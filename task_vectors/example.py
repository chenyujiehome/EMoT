from task_vectors import TaskVector
pretrained_checkpoint="/home/yujie/SuPreM/task_vectors/checkpoints/supervised_suprem_segresnet_2100.pth"
finetuned_checkpoint_fold_1="/home/yujie/SuPreM/task_vectors/checkpoints/best_model_fold_1.pth"
finetuned_checkpoint_fold_3="/home/yujie/SuPreM/task_vectors/checkpoints/best_model_fold_3.pth"
task_vector_A = TaskVector(pretrained_checkpoint, finetuned_checkpoint_fold_1)
task_vector_B = TaskVector(pretrained_checkpoint, finetuned_checkpoint_fold_3)
vector_C = task_vector_B + task_vector_A
vector_D= task_vector_B + task_vector_A
print(task_vector_A)