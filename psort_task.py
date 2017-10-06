import tensorflow as tf
import numpy as np
import argparse
from model import NTMPrioritySortModel
#from utils import generate_random_strings
from utils import generate_psort_data
import matplotlib.pyplot as plt
import os

task_name = "psort_task"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--test_seq_length', type=int, default=20)
    parser.add_argument('--model', default="NTM")
    parser.add_argument('--rnn_size', default=128)
    parser.add_argument('--rnn_num_layers', default=3)
    parser.add_argument('--max_seq_length', default=4)
    parser.add_argument('--memory_size', default=128)
    parser.add_argument('--memory_vector_dim', default=20)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--vector_dim', default=9)
    parser.add_argument('--shift_range', default=1)
    parser.add_argument('--num_epoches', default=1000000)
    parser.add_argument('--learning_rate', default=3*1e-5)
    parser.add_argument('--save_dir', default='./save/%s' % task_name )
    parser.add_argument('--tensorboard_dir', default='./summary/%s' % task_name )
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
#    model_list = [NTMCopyModel(args, 1)]
#    for seq_length in range(2, args.max_seq_length + 1):
#        print ( "append model list %s/%s" %(seq_length, args.max_seq_length)  )
#        model_list.append(NTMCopyModel(args, seq_length, reuse=True))
    model_list = [NTMPrioritySortModel(args, args.max_seq_length)]
    # model = NTM_model(args, args.max_seq_length)
 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1 

    with tf.Session(config=config) as sess:
        if args.restore_training:
            saver = tf.train.Saver( max_to_keep=None )
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
        plt.ion()
        plt.show()
        for b in range(args.num_epoches):
            #seq_length = np.random.randint(1, args.max_seq_length + 1)
            #model = model_list[seq_length - 1]
            seq_length = args.max_seq_length
            model = model_list[0]
            # seq_length = args.max_seq_length
            x, y = generate_psort_data(args.batch_size, seq_length, args.vector_dim)
            feed_dict = {model.x: x, model.y: y}
            # print(sess.run([model.state_list, model.output_list], feed_dict=feed_dict))
            if b % 100 == 0:        # test
                p = 0               # select p th sample in the batch to show
                print(x[p, :, :])
                print(y[p, :, :])
                sess.run(model.o, feed_dict=feed_dict)[p, :, :]
                state_list = sess.run(model.state_list, feed_dict=feed_dict)
#                if args.model == 'NTM':
#                    w_plot = []
#                    M_plot = np.concatenate([state['M'][p, :, :] for state in state_list])
#                    for state in state_list:
#                        w_plot.append(np.concatenate([state['w_list'][0][p, :], state['w_list'][1][p, :]]))
#                    plt.imshow(w_plot, interpolation='nearest', cmap='gray')
#                    plt.draw()
#                    plt.pause(0.001)
                copy_loss = sess.run(model.copy_loss, feed_dict=feed_dict)
                merged_summary = sess.run(model.copy_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)
                print('batches %d, loss %g' % (b, copy_loss))
            else:                   # train
                sess.run(model.train_op, feed_dict=feed_dict)
            if b % 5000 == 0 and b > 0:
                if not os.path.exists(args.save_dir + '/' + args.model):
                    os.makedirs(args.save_dir + '/' + args.model)
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)


def test(args):
    model = NTMCopyModel(args, args.test_seq_length)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # generate data (batch_size, test_seq_length, vec_dim)
        x = generate_random_strings(args.batch_size, args.test_seq_length, args.vector_dim)
        feed_dict = {model.x: x}
        # run model
        output, copy_loss, state_list = sess.run( [model.o, model.copy_loss, model.state_list], feed_dict=feed_dict )
        # for each batch, print input and output
        for p in range(args.batch_size):
            print(x[p, :, :])
            print(output[p, :, :])
        # print total loss in this batch_size
        print('copy_loss: %g' % copy_loss)
        # plot NTM state
        if args.model == 'NTM':
            w_plot = []
            for state in state_list:
                w_plot.append(np.concatenate([state['w_list'][0][p, :], state['w_list'][1][p, :]]))
            plt.imshow(w_plot, interpolation='nearest', cmap='gray')
            plt.show()

if __name__ == '__main__':
    main()
