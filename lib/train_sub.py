import tensorflow as tf
import trnmodel
import datareader as dr
import config_sub as config
import os
import numpy as np
import time

from timeit import default_timer
from contextlib import contextmanager


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def do_validation(m_valid, sess, valid_path):

    # dataset reader setting #

    valid_dr = dr.DataReader(valid_path["valid_input_path"], valid_path["valid_output_path"],
                             valid_path["norm_path"], dist_num=config.dist_num)

    avg_valid_accuracy = 0.
    avg_valid_cost = 0.
    itr_sum = 0.

    accuracy_list = [0 for i in range(valid_dr._file_len)]
    cost_list = [0 for i in range(valid_dr._file_len)]
    itr_file = 0

    while True:

        valid_inputs, valid_labels = valid_dr.next_batch(config.batch_size)

        if valid_dr.file_change_checker():
            # print(itr_file)
            accuracy_list[itr_file] = avg_valid_accuracy / itr_sum
            cost_list[itr_file] = avg_valid_cost / itr_sum
            avg_valid_cost = 0.
            avg_valid_accuracy = 0.
            itr_sum = 0
            itr_file += 1
            valid_dr.file_change_initialize()

        if valid_dr.eof_checker():
            valid_dr.reader_initialize()
            print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
            break

        feed_dict = {m_valid.inputs: valid_inputs, m_valid.labels: valid_labels}

        # valid_cost, valid_softpred, valid_raw_labels\
        #     = sess.run([m_valid.cost, m_valid.softpred, m_valid.raw_labels], feed_dict=feed_dict)
        #
        # fpr, tpr, thresholds = metrics.roc_curve(valid_raw_labels, valid_softpred, pos_label=1)
        # valid_auc = metrics.auc(fpr, tpr)

        valid_cost, valid_accuracy = sess.run([m_valid.Tcost, m_valid.accuracy], feed_dict=feed_dict)

        avg_valid_accuracy += valid_accuracy
        avg_valid_cost += valid_cost
        itr_sum += 1

    total_avg_valid_accuracy = np.asscalar(np.mean(np.asarray(accuracy_list)))
    total_avg_valid_cost = np.asscalar(np.mean(np.asarray(cost_list)))

    return total_avg_valid_accuracy, total_avg_valid_cost


def do_validation_G(m_valid, sess, valid_path):

    # dataset reader setting #

    valid_dr = dr.DataReader(valid_path["valid_input_path"], valid_path["valid_output_path"],
                             valid_path["norm_path"], dist_num=config.dist_num)

    avg_valid_accuracy = 0.
    avg_valid_cost = 0.
    itr_sum = 0.

    accuracy_list = [0 for i in range(valid_dr._file_len)]
    cost_list = [0 for i in range(valid_dr._file_len)]
    itr_file = 0

    while True:

        valid_inputs, valid_labels = valid_dr.next_batch(config.batch_size)

        if valid_dr.file_change_checker():
            # print(itr_file)
            accuracy_list[itr_file] = avg_valid_accuracy / itr_sum
            cost_list[itr_file] = avg_valid_cost / itr_sum
            avg_valid_cost = 0.
            avg_valid_accuracy = 0.
            itr_sum = 0
            itr_file += 1
            valid_dr.file_change_initialize()

        if valid_dr.eof_checker():
            valid_dr.reader_initialize()
            print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
            break

        feed_dict = {m_valid.inputs: valid_inputs, m_valid.labels: valid_labels}

        # valid_cost, valid_softpred, valid_raw_labels\
        #     = sess.run([m_valid.cost, m_valid.softpred, m_valid.raw_labels], feed_dict=feed_dict)
        #
        # fpr, tpr, thresholds = metrics.roc_curve(valid_raw_labels, valid_softpred, pos_label=1)
        # valid_auc = metrics.auc(fpr, tpr)

        valid_cost, valid_accuracy = sess.run([m_valid.C1_loss, m_valid.d1_loss], feed_dict=feed_dict)

        avg_valid_accuracy += valid_accuracy
        avg_valid_cost += valid_cost
        itr_sum += 1

    total_avg_valid_accuracy = np.asscalar(np.mean(np.asarray(accuracy_list)))
    total_avg_valid_cost = np.asscalar(np.mean(np.asarray(cost_list)))

    return total_avg_valid_accuracy, total_avg_valid_cost


def do_validationS(m_valid, sess, valid_path):

    # dataset reader setting #

    valid_dr = dr.DataReader(valid_path["valid_input_path"], valid_path["valid_output_path"],
                             valid_path["norm_path"], dist_num=config.dist_num)

    avg_valid_accuracy = 0.
    avg_valid_cost = 0.
    itr_sum = 0.

    accuracy_list = [0 for i in range(valid_dr._file_len)]
    cost_list = [0 for i in range(valid_dr._file_len)]
    itr_file = 0

    while True:

        valid_inputs, valid_labels = valid_dr.next_batch(config.batch_size)

        if valid_dr.file_change_checker():
            # print(itr_file)
            accuracy_list[itr_file] = avg_valid_accuracy / itr_sum
            cost_list[itr_file] = avg_valid_cost / itr_sum
            avg_valid_cost = 0.
            avg_valid_accuracy = 0.
            itr_sum = 0
            itr_file += 1
            valid_dr.file_change_initialize()

        if valid_dr.eof_checker():
            valid_dr.reader_initialize()
            print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
            break

        feed_dict = {m_valid.inputs: valid_inputs, m_valid.labels: valid_labels}

        # valid_cost, valid_softpred, valid_raw_labels\
        #     = sess.run([m_valid.cost, m_valid.softpred, m_valid.raw_labels], feed_dict=feed_dict)
        #
        # fpr, tpr, thresholds = metrics.roc_curve(valid_raw_labels, valid_softpred, pos_label=1)
        # valid_auc = metrics.auc(fpr, tpr)

        valid_cost, valid_accuracy = sess.run([m_valid.S_loss, m_valid.accuracy2], feed_dict=feed_dict)

        avg_valid_accuracy += valid_accuracy
        avg_valid_cost += valid_cost
        itr_sum += 1

    total_avg_valid_accuracy = np.asscalar(np.mean(np.asarray(accuracy_list)))
    total_avg_valid_cost = np.asscalar(np.mean(np.asarray(cost_list)))

    return total_avg_valid_accuracy, total_avg_valid_cost


def main(argv=None):
    mode = 2
    #mode 0 : train_teacher, 1: student1, 2:student2, 3:student3, 4:student:hidden
    # set train path

    if argv is None:

        train_input_path = os.path.abspath('../data/train/wav')
        train_output_path = os.path.abspath('../data/train/lab')
        norm_path = os.path.abspath('../data/train/norm')

        # set valid path
        valid_input_path = os.path.abspath('../data/valid/wav')
        valid_output_path = os.path.abspath('../data/valid/lab')
        logs_dir = os.path.abspath('../logs')
        logs_dir_student = os.path.abspath('../logs_student')
        logs_dir_student_total = os.path.abspath('../logs_student_total')
    else:
        train_input_path = argv + '/data/train/wav'
        train_output_path = argv + '/data/train/lab'
        norm_path = argv + '/data/train/norm'

        # set valid path
        valid_input_path = argv + '/data/valid/wav'
        valid_output_path = argv + '/data/valid/lab'
        logs_dir = argv + '/logs'
        logs_dir_student = argv + '/logs_student'
        logs_dir_student_total = argv + '/logs_student_total'
    #                               Graph Part                               #

    print("Graph initialization...")

    global_step = tf.Variable(0, trainable=False)

    with tf.device(config.device):
        with tf.variable_scope("model", reuse=None):
            m_train = trnmodel.Model(is_training=True, global_step=global_step,keep_prob=0.5,mode=mode)
        with tf.variable_scope("model", reuse=True):
            m_valid = trnmodel.Model(is_training=False, global_step=global_step,keep_prob=1.0,mode=mode)

    print("Done")

    #                               Summary Part                             #

    tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    with open(logs_dir + "/tensor_names.txt", 'w') as f:
        for t_name in tensor_names:
            f.write("%s\n" % str(t_name))

    print("Setting up summary op...")

    summary_ph = tf.placeholder(dtype=tf.float32)
    with tf.variable_scope("Training_procedure"):

        cost_summary_op = tf.summary.scalar("cost", summary_ph)
        accuracy_summary_op = tf.summary.scalar("accuracy", summary_ph)

    print("Done")

    #                               Model Save Part                           #

    print("Setting up Saver...")
    list_var_teacher = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="model/teacher")
    list_var_student = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="model/student/sbase")
    saver = tf.train.Saver(var_list=list_var_teacher)
    saver_student = tf.train.Saver(var_list=list_var_student)
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    ckpt_student = tf.train.get_checkpoint_state(logs_dir_student)
    print("Done")

    #                               Session Part                              #

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    train_summary_writer = tf.summary.FileWriter(logs_dir + '/train/', sess.graph, max_queue=2)
    valid_summary_writer = tf.summary.FileWriter(logs_dir + '/valid/', max_queue=2)

    sess.run(tf.global_variables_initializer()) # for all variable initialization

    if ckpt and ckpt.model_checkpoint_path:  # model restore

        print("Model restored...")

        saver.restore(sess, ckpt.model_checkpoint_path)

        print("Done")
    # if ckpt_student and ckpt_student.model_checkpoint_path:  # model restore
    #
    #     print("Model restored...")
    #
    #     saver_student.restore(sess, ckpt_student.model_checkpoint_path)
    #
    #     print("Done")
    # else:
    #     sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization

    # datareader initialization
    train_dr = dr.DataReader(train_input_path, train_output_path, norm_path, dist_num=config.dist_num)
    # valid_dr = dr.DataReader(valid_input_path, valid_output_path, norm_path, dist_num=config.dist_num)
    valid_path = {'valid_input_path': valid_input_path, 'valid_output_path': valid_output_path, 'norm_path': norm_path}

    if mode == 0:
        for itr in range(config.max_epoch):

            start_time = time.time()
            train_inputs, train_labels = train_dr.next_batch(config.batch_size)
            feed_dict = {m_train.inputs: train_inputs, m_train.labels: train_labels}
            sess.run(m_train.train_op, feed_dict=feed_dict)
            elapsed_time = time.time() - start_time

            # print("time_per_step:%.4f" % elapsed_time)

            if itr % 10 == 0 and itr >= 0:

                train_cost, train_accuracy \
                    = sess.run([m_train.Tcost, m_train.accuracy], feed_dict=feed_dict)

                # print("Step: %d, train_cost: %.4f, train_accuracy=%4.4f, train_time=%.4f"
                #       % (itr, train_cost, train_accuracy * 100, el_tim))
                print("Step: %d, train_cost: %.4f, train_accuracy=%4.4f"
                      % (itr, train_cost, train_accuracy * 100))
                train_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: train_cost})
                train_accuracy_summary_str = sess.run(accuracy_summary_op, feed_dict={summary_ph: train_accuracy})
                train_summary_writer.add_summary(train_cost_summary_str,
                                                 itr)  # write the train phase summary to event files
                train_summary_writer.add_summary(train_accuracy_summary_str, itr)

            if itr % config.val_step == 0 and itr > 0:

                saver.save(sess, logs_dir + "/model.ckpt", itr)  # model save
                print('validation start!')
                valid_accuracy, valid_cost = do_validation(m_valid, sess, valid_path)

                print("valid_cost: %.4f, valid_accuracy=%4.4f" % (valid_cost, valid_accuracy * 100))
                valid_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: valid_cost})
                valid_accuracy_summary_str = sess.run(accuracy_summary_op, feed_dict={summary_ph: valid_accuracy})
                valid_summary_writer.add_summary(valid_cost_summary_str,
                                                 itr)  # write the train phase summary to event files
                valid_summary_writer.add_summary(valid_accuracy_summary_str, itr)

    elif mode == 1:
        for itr in range(config.max_epoch):

            start_time = time.time()
            train_inputs, train_labels = train_dr.next_batch(config.batch_size)
            feed_dict = {m_train.inputs: train_inputs, m_train.labels: train_labels}
            sess.run(m_train.train_op2, feed_dict=feed_dict)
            elapsed_time = time.time() - start_time

            if itr % 10 == 0 and itr >= 0:

                train_C1_loss, train_D_loss \
                    = sess.run([m_train.C1_loss, m_train.d1_loss], feed_dict=feed_dict)

                # print("Step: %d, train_cost: %.4f, train_accuracy=%4.4f, train_time=%.4f"
                #       % (itr, train_cost, train_accuracy * 100, el_tim))
                print("Step: %d, train_G_loss: %f, train_D_loss=%f"
                      % (itr, train_C1_loss*1000, train_D_loss))
                # train_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: train_cost})
                # train_accuracy_summary_str = sess.run(accuracy_summary_op, feed_dict={summary_ph: train_accuracy})
                # train_summary_writer.add_summary(train_cost_summary_str,
                #                                  itr)  # write the train phase summary to event files
                # train_summary_writer.add_summary(train_accuracy_summary_str, itr)

            if itr % config.val_step == 0 and itr > 0:
                saver_student.save(sess, logs_dir_student + "/model.ckpt", itr)  # model save
                print('validation part!')
                # valid_accuracy, valid_cost = do_validation_G(m_valid, sess, valid_path)
                #
                # print("C_loss: %.4f, d1_loss=%4.4f" % (valid_cost, valid_accuracy * 100))
                # valid_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: valid_cost})
                # valid_accuracy_summary_str = sess.run(accuracy_summary_op, feed_dict={summary_ph: valid_accuracy})
                # valid_summary_writer.add_summary(valid_cost_summary_str,
                #                                  itr)  # write the train phase summary to event files
                # valid_summary_writer.add_summary(valid_accuracy_summary_str, itr)
    elif mode == 2:
        for itr in range(config.max_epoch):

            start_time = time.time()
            train_inputs, train_labels = train_dr.next_batch(config.batch_size)
            feed_dict = {m_train.inputs: train_inputs, m_train.labels: train_labels}
            sess.run(m_train.train_op3, feed_dict=feed_dict)
            elapsed_time = time.time() - start_time

            if itr % 10 == 0 and itr >= 0:

                train_cost, train_accuracy \
                    = sess.run([m_train.S_loss, m_train.accuracy2], feed_dict=feed_dict)

                # print("Step: %d, train_cost: %.4f, train_accuracy=%4.4f, train_time=%.4f"
                #       % (itr, train_cost, train_accuracy * 100, el_tim))
                print("Step: %d, train_cost: %.4f, train_accuracy=%4.4f"
                      % (itr, train_cost, train_accuracy * 100))
                train_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: train_cost})
                train_accuracy_summary_str = sess.run(accuracy_summary_op, feed_dict={summary_ph: train_accuracy})
                train_summary_writer.add_summary(train_cost_summary_str,
                                                 itr)  # write the train phase summary to event files
                train_summary_writer.add_summary(train_accuracy_summary_str, itr)

            if itr % config.val_step == 0 and itr > 0:

                saver.save(sess, logs_dir_student_total + "/model.ckpt", itr)  # model save
                print('validation start!')
                valid_accuracy, valid_cost = do_validationS(m_valid, sess, valid_path)

                print("valid_cost: %.4f, valid_accuracy=%4.4f" % (valid_cost, valid_accuracy * 100))
                # valid_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: valid_cost})
                # valid_accuracy_summary_str = sess.run(accuracy_summary_op, feed_dict={summary_ph: valid_accuracy})
                # valid_summary_writer.add_summary(valid_cost_summary_str,
                #                                  itr)  # write the train phase summary to event files
                # valid_summary_writer.add_summary(valid_accuracy_summary_str, itr)

if __name__ == "__main__":
    # tf.app.run()
    main()
