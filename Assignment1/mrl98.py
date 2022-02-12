import numpy as np


class DataStreamer():

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
        if current_idx * self.batch_size >= self.stream_len:
            raise StopIteration

        self.current_idx += 1
        return self.data[self.current_idx:self.current_idx + self.batch_size]

    def generate_data(self, data_len):
        return np.random.randint(data_len, size=data_len)


class Buffer:

    def __init__(self, pos=0, k=3):

        self.k = k
        self.pos = pos
        self.data = np.empty((self.k, ))
        self.weight = 0
        self.level = 0  # the bffer's level in the tree
        self.full = False  # whther the buffer is empty

    def fill(self, data):
        """fill the buffer with data, data should be of shape (k, )"""
        self.data = data
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
        i = 0
        self.n_empty_buffers = b
        self.full_buffers = []

    def collapse_policy(self, data_stream):
        """Running the algorithm proposed in the paper and merge nodes from left to right"""

        while self.n_empty_buffers > 0:
            num_full_buffers = len(self.full_buffers)
            l = min([buffer.level for buffer in self.full_buffers]
                    ) if num_full_buffers > 0 else 0
            if self.n_empty_buffers == 1:
                empty_buffer = Buffer(k=3)
                empty_buffer.level = l
                self.new(empty_buffer, next(data_stream))
            elif self.n_empty_buffers >= 2:
                for i in range(self.n_empty_buffers):
                    empty_buffer = Buffer(k=3)
                    empty_buffer.level = 0
                    self.new(empty_buffer, next(data_stream))
            else:
                buffers_to_merge = self.get_buffers_at_level(l)
                output_buffer = self.collapse(buffers_to_merge)
                self.full_buffers.append(output_buffer)
                # self.empty_buffers.

    def get_buffers_at_level(self, level):
        """Get the buffers at the given level"""

        res_buffers = []
        for buffer in self.buffers:
            if buffer.level == level:
                res_buffers.append(buffer)
        return res_buffers

    def collapse(self, buffers):
        """Collapse the given buffers into one buffer, this will leave one fill buffer (the output buffer), 
        and the rest will be empty"""

        output_buffer = Buffer(buffers[0].pos, k=self.k)
        output_data = []
        output_weight = np.sum([buffer.weight for buffer in buffers])
        output_buffer.weight = output_weight
        current_idx = 0
        output_idx = 0
        indices = [0] * len(buffers)
        num_values = output_weight * self.k
        while current_idx < num_values:
            current_items = []
            for i in range(len(buffers)):
                item = buffers[i][indices[i]]
                current_items.append((item, i))
            current_min_item = min(current_items)
            current_idx += current_min_item[1]

            if current_idx % (output_idx * output_weight + int((output_weight + 1)/2)):
                print('hitting a desired position')
                output_idx += 1
                output_data.append(current_min_item[0])
        assert len(output_data) == self.k
        output_buffer.fill(np.array(output_data))
        # mark the empty buffers as empty
        self.fix_buffers_positions(output_buffer)
        return output_buffer

    def fix_buffers_positions(self, output_buffer):
        for buffer in self.full_buffers:
            if buffer.pos != output_buffer.pos:
                del self.full_buffers[buffer.pos]
                self.n_empty_buffers += 1
        for i in range(len(self.full_buffers)):
            self.full_buffers[i].pos = i

    def new(self, buffer, data):
        """Fill the given buffer with data and assign its weight to 1"""
        buffer.fill(data)
        buffer.weight = 1
        buffer.pos = len(self.full_buffers)
        self.full_buffers.append(buffer)

    def output(self):
        pass


def main():
    streamer = DataStreamer(10, 5)


if __name__ == "__main__":
    main()
