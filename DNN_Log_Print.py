import DNN_tools


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    if R_dic['input_dim'] == 1:
        if R_dic['PDE_type'] == 'pLaplace':
            DNN_tools.log_string('The order of pLaplace operator: %s\n' % (R_dic['order2pLaplace_operator']),
                                 log_fileout)
            DNN_tools.log_string('The epsilon to pLaplace operator: %f\n' % (R_dic['epsilon']), log_fileout)
        if R_dic['PDE_type'] == 'Possion_Boltzmann':
            DNN_tools.log_string('The order of pLaplace operator: %s\n' % (R_dic['order2pLaplace_operator']),
                                 log_fileout)
            DNN_tools.log_string('The epsilon to pLaplace operator: %f\n' % (R_dic['epsilon']), log_fileout)
    elif R_dic['input_dim'] == 2:
        if R_dic['PDE_type'] == 'pLaplace_implicit' or R_dic['PDE_type'] == 'pLaplace_explicit':
            DNN_tools.log_string('The order of pLaplace operator: %s\n' % (R_dic['order2pLaplace_operator']), log_fileout)
            DNN_tools.log_string('The mesh_number: %f\n' % (R_dic['mesh_number']), log_fileout)
        elif R_dic['PDE_type'] == 'Possion_Boltzmann':
            DNN_tools.log_string('The mesh_number: %f\n' % (R_dic['mesh_number']), log_fileout)
    else:
        DNN_tools.log_string('The order of pLaplace operator: %s\n' % (R_dic['order2pLaplace_operator']),
                             log_fileout)
        DNN_tools.log_string('The epsilon to pLaplace operator: %f\n' % (R_dic['epsilon']), log_fileout)

    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model2NN']), log_fileout)
    if R_dic['model2NN'] == 'DNN_FourierBase' or R_dic['model2NN'] == 'Fourier_DNN' or \
            R_dic['model2NN'] == 'Fourier_SubDNN':
        DNN_tools.log_string('Activate function for NN-input: %s\n' % '[Sin;Cos]', log_fileout)
    else:
        DNN_tools.log_string('Activate function for NN-input: %s\n' % str(R_dic['name2act_in']), log_fileout)
    DNN_tools.log_string('Activate function for NN-hidden: %s\n' % str(R_dic['name2act_hidden']), log_fileout)
    DNN_tools.log_string('Activate function for NN-output: %s\n' % str(R_dic['name2act_out']), log_fileout)
    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)
    if R_dic['model2NN'] != 'DNN':
        DNN_tools.log_string('The frequency to neural network: %s\n' % (R_dic['freq']), log_fileout)

    if R_dic['model2NN'] == 'DNN_FourierBase' or R_dic['model2NN'] == 'Fourier_DNN' or \
            R_dic['model2NN'] == 'Fourier_SubDNN':
        DNN_tools.log_string('The scale-factor to fourier basis: %s\n' % (R_dic['sfourier']), log_fileout)

    if R_dic['loss_type'] == 'variational_loss':
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: L2 loss\n', log_fileout)

    if (R_dic['train_model']) == 'union_training':
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss', log_fileout)
    elif (R_dic['train_model']) == 'group3_training':
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_it + loss_bd', log_fileout)
    elif (R_dic['train_model']) == 'group2_training':
        DNN_tools.log_string('The model for training loss: %s\n' % 'total loss + loss_bd', log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R_dic['activate_penalty2bd_increase'] == 1:
        DNN_tools.log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R_dic['activate_penalty2bd_increase'] == 2:
        DNN_tools.log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        DNN_tools.log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


def print_and_log_train_one_epoch(i_epoch, run_time, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp, loss_bd_tmp, loss_tmp,
                                  train_mse_tmp, train_rel_tmp, log_out=None):
    # 将运行结果打印出来
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('boundary penalty: %.10f' % temp_penalty_bd)
    print('weights and biases with  penalty: %.10f' % pwb)
    print('loss_it for training: %.10f' % loss_it_tmp)
    print('loss_bd for training: %.10f' % loss_bd_tmp)
    print('loss for training: %.10f' % loss_tmp)
    print('solution mean square error for training: %.10f' % train_mse_tmp)
    print('solution residual error for training: %.10f\n' % train_rel_tmp)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %.10f' % tmp_lr, log_out)
    DNN_tools.log_string('boundary penalty: %.10f' % temp_penalty_bd, log_out)
    DNN_tools.log_string('weights and biases with  penalty: %.10f' % pwb, log_out)
    DNN_tools.log_string('loss_it for training: %.10f' % loss_it_tmp, log_out)
    DNN_tools.log_string('loss_bd for training: %.10f' % loss_bd_tmp, log_out)
    DNN_tools.log_string('loss for training: %.10f' % loss_tmp, log_out)
    DNN_tools.log_string('solution mean square error for training: %.10f' % train_mse_tmp, log_out)
    DNN_tools.log_string('solution residual error for training: %.10f\n' % train_rel_tmp, log_out)


def print_and_log_test_one_epoch(mse2test, res2test, log_out=None):
    # 将运行结果打印出来
    print('mean square error of predict and real for testing: %.10f' % mse2test)
    print('residual error of predict and real for testing: %.10f\n' % res2test)

    DNN_tools.log_string('mean square error of predict and real for testing: %.10f' % mse2test, log_out)
    DNN_tools.log_string('residual error of predict and real for testing: %.10f\n\n' % res2test, log_out)