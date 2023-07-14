clc;
clear all
close all
M=101;

original_index = int32(linspace(0,M*M-1,M*M));
disorder_index = reorder_index(original_index, M);

order_index = recover_index(disorder_index, M);

if M==101
    save('disorder_index101.mat','disorder_index');
elseif M==51
    save('disorder_index51.mat','disorder_index');
end