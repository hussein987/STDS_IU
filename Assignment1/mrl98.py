import numpy as np
import warnings
import argparse
import time
import sys

class DataStreamer():
    '''
        Simulates a stream of data
    '''

    def __init__(self, stream_len, batch_size):

        if stream_len % batch_size != 0:
            raise ValueError(
                'The stream length must be divisible by the batch size')

        self.stream_len = stream_len
        self.batch_size = batch_size
        self.data = self.generate_data(self.stream_len)
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):

        current_idx = self.current_idx
        if current_idx >= self.stream_len:
            raise StopIteration

        self.current_idx += self.batch_size
        batch = self.data[current_idx: current_idx + self.batch_size]
        return batch

    def done(self):
        return self.current_idx >= self.stream_len

    def generate_data(self, data_len):
        return np.random.randint(data_len, size=data_len)


class Buffer:

    def __init__(self, k=5):

        self.k = k
        self.data = np.empty((self.k, ))
        self.weight = 0
        self.level = 0  # the bffer's level in the tree
        self.full = False  # whther the buffer is empty

    def fill(self, data):
        """fill the buffer with data, data should be of shape (k, )"""
        self.data = np.sort(data)
        self.full = True

    def upgrade(self):
        self.level += 1

    def __getitem__(self, idx):
        return self.data[idx]


class MRL98():
    """The main class of the algorithm"""

    def __init__(self, N=1e3, b=3, k=5):
        self.N = N
        self.b = b
        self.k = k
        self.C = 0  # number of COLLAPSE operation carried out
        self.W = 0  # sum of weights of the output buffers in collapse operations
        self.n_empty_buffers = b
        self.full_buffers = []
        self.memory_usage = 0

    # Verified to work
    def run_collapse_policy(self, data_stream, phi):
        """Running the algorithm proposed in the paper and merge nodes from left to right"""
        start_time = time.time()
        while not data_stream.done():
            # print(f'current streamer index: {data_stream.current_idx}')
            num_full_buffers = len(self.full_buffers)
            l = min([buffer.level for buffer in self.full_buffers]
                    ) if num_full_buffers > 0 else 0
            # If there's exactly one empty buffer, invoke NEW and assign it level l
            if self.n_empty_buffers == 1:
                empty_buffer = Buffer(k=self.k)
                empty_buffer.level = l
                self.new(empty_buffer, next(data_stream))
                self.n_empty_buffers = 0
            # If there are at least two empty buffers, invoke NEW on each and assign level 0 to each one
            elif self.n_empty_buffers >= 2:
                for i in range(self.n_empty_buffers):
                    empty_buffer = Buffer(k=self.k)
                    empty_buffer.level = 0
                    self.new(empty_buffer, next(data_stream))
                self.n_empty_buffers = 0
            # If there are no empty buffers, invoke COLLAPSE on the set of buffers at level l,
            # assing the output buffer, level l+1
            else:
                if not self.memory_usage:
                    from pympler import asizeof
                    # self.memory_usage = sys.getsizeof(self.full_buffers)
                    self.memory_usage = asizeof.asizeof(self.full_buffers)
                buffers_to_merge = self.get_buffers_at_level(l)
                self.collapse(buffers_to_merge, l)
                self.n_empty_buffers = len(buffers_to_merge) - 1

        # There's no more data, invoke Output
        output = self.output(self.full_buffers, phi)
        self.time_elapsed = time.time() - start_time
        return output


    def get_buffers_at_level(self, level):
        '''
            Get the buffers at the given level.
            NOTE: pop the buffers that needs to be merged from the list of full_buffers
        '''
        all_full_buffers = self.full_buffers
        self.full_buffers = []

        res_buffers = []
        for buffer in all_full_buffers:
            if buffer.level == level:
                res_buffers.append(buffer)
            else:
                self.full_buffers.append(buffer)
        return res_buffers

    # Verified to work
    def collapse(self, buffers, l):
        """Collapse the given buffers into one buffer, this will leave one fill buffer (the output buffer), 
        and the rest will be empty"""

        # print(f'number of buffers: {len(buffers)}')
        output_buffer = Buffer(k=self.k)
        output_data = []
        output_weight = np.sum([buffer.weight for buffer in buffers])
        output_buffer.weight = output_weight
        current_idx, output_idx = 0, 0
        indices = [0] * len(buffers)
        num_values = output_weight * self.k
        while current_idx < num_values:
            current_items = []
            for i in range(len(buffers)):
                if indices[i] < self.k:
                    item = buffers[i][indices[i]]
                    current_items.append((item, i))
            current_min_item = min(current_items)
            indices[current_min_item[1]] += 1
            current_idx += buffers[current_min_item[1]].weight
            # print(f'current item: {current_items}, current index: {current_idx}')

            critical_position = (
                output_idx * output_weight + int((output_weight + 1)/2))
            if current_idx >= critical_position:
                # print('hitting a desired position')
                output_idx += 1
                output_data.append(current_min_item[0])
        assert len(output_data) == self.k
        output_buffer.fill(np.array(output_data))
        output_buffer.level = l + 1
        self.full_buffers.append(output_buffer)
        # print(f'output buffer weight: {output_buffer.weight}')
        return output_buffer

    # Verified to work
    def new(self, buffer, data):
        """Fill the given buffer with data and assign its weight to 1"""
        buffer.fill(data)
        buffer.weight = 1
        self.full_buffers.append(buffer)

    # Verified to work
    def output(self, buffers, phi):
        '''
            OUTPUT takes c >= 2 full input buffers, X_1, X_2, ... , X_c, of size k,
            and returns a single element, corresponding to the approximate phi'-quantile
            of the augmented dataset. The phi-quantile of the original dataset corresponds to the
            phi'-quantile of the augmented dataset, which consists of the original elements  plus
            the -inf and +inf elements added to the last buffer.
            NOTE: OUTPUT is invoked on the final set of full buffers.
        '''
        W = np.sum([buffer.weight for buffer in buffers]
                   )  # Sum of the input buffers' weights
        output_element_position = np.ceil(phi * self.k * W)

        current_idx = 0
        indices = [0] * len(buffers)
        num_values = W * self.k
        while current_idx < num_values:
            current_items = []
            for i in range(len(buffers)):
                if indices[i] < self.k:
                    item = buffers[i][indices[i]]
                    current_items.append((item, i))
            current_min_item = min(current_items)
            indices[current_min_item[1]] += 1
            current_idx += buffers[current_min_item[1]].weight

            if current_idx >= output_element_position:
                print('reached the desired element')
                return current_min_item[0]
        warnings.warn("Couldn't find the desired element")

def main(args):
    data_stream = DataStreamer(args.data_size, args.batch_size)
    mrl = MRL98(b=args.num_buffers, k=args.items_per_buffer)
    output = mrl.run_collapse_policy(data_stream, args.phi)
    print(f'the {args.phi}-th quantile is: {output}')


def get_args():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-k', '--items_per_buffer', default=3, type=int,
                        help='number of items per buffer')
    parser.add_argument('-b', '--num_buffers', default=5, type=int,
                        help='number of buffers')
    parser.add_argument('--data_size', default=3000, type=int,
                        help='number of items per buffer')
    parser.add_argument('--batch_size', default=3, type=int,
                        help='number of items per buffer')
    parser.add_argument('--phi', default=0.5, type=int,
    help='number of items per buffer')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
