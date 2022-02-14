from mrl98 import *
import numpy as np
import os, psutil
import time
import matplotlib.pyplot as plt

def get_data_size(batch_size, buffer_size):
    data_size = batch_size * (int(np.sum([i * (buffer_size - i + 1) for i in range(1, buffer_size+1)])) - 1)
    print(data_size)
    return data_size 

def main(args):
    exp_dict = dict()
    exp_dict['k'] = [3, 5, 10, 15, 20, 25, 35, 40, 50, 58, 59, 60, 77, 77, 77, 77]
    exp_dict['b'] = [3, 5, 7, 9, 10, 11, 15, 17, 18, 20, 21, 22, 23, 25, 26, 27]
    exp_dict['data_size'] = [get_data_size(batch_sz, buffer_sz) for batch_sz, buffer_sz in zip(exp_dict['k'], exp_dict['b'])]
    exp_dict['batch_size'] = exp_dict['k']
    times = []
    memories = []
    for i in range(len(exp_dict['k'])):
        k = exp_dict['k'][i]
        b = exp_dict['b'][i]
        data_size = exp_dict['data_size'][i]
        batch_size = exp_dict['batch_size'][i]
        print(f'performing expriment with k = {k}, b={b}, data_size = {data_size}, batch_size = {batch_size}')
        data_stream = DataStreamer(data_size, batch_size)
        mrl = MRL98(b=args.num_buffers, k=args.items_per_buffer)
        output = mrl.run_collapse_policy(data_stream, args.phi)
        memory_usage = mrl.memory_usage
        time_elapsed = mrl.time_elapsed
        times.append(time_elapsed)
        memories.append(memory_usage)
        print(f'the {args.phi}-th quantile is: {output}')
        print(f'Memory usage: {memory_usage} MB')
        print(f'Execution time: {time_elapsed} seconds')
        print('='*100)
    
    print(exp_dict['data_size'])
    print(times)
    plt.plot(exp_dict['data_size'], times, '-gD')
    plt.ylabel('tme (sec)')
    plt.xlabel('N')
    plt.title('execution time')
    plt.grid()
    plt.savefig('execution_time.png')
    plt.show()

    # report memory
    # plt.plot(exp_dict['data_size'], times, '-rD')
    # plt.ylabel('memory (KB)')
    # plt.xlabel('N')
    # plt.title('memory usage')
    # plt.grid()
    # # plt.savefig('memory_usage.png')
    # plt.show()

    



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