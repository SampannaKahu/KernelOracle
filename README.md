### KernelOracle: Predicting the Linux Scheduler's Next Move with Deep Learning
Efficient task scheduling is paramount in the Linux kernel, where the Completely Fair Scheduler (CFS) meticulously manages CPU resources to balance high utilization with interactive responsiveness. This research pioneers the use of deep learning techniques to predict the sequence of tasks selected by CFS, aiming to evaluate the feasibility of a more generalized and potentially more adaptive task scheduler for diverse workloads. Our core contributions are twofold: first, the systematic generation and curation of a novel scheduling dataset from a running Linux kernel, capturing real-world CFS behavior; and second, the development, training, and evaluation of a Long Short-Term Memory (LSTM) network designed to accurately forecast the next task to be scheduled. Our paper further discusses the practical pathways and implications of integrating such a predictive model into the kernel's scheduling framework. The findings and methodologies presented herein open avenues for data-driven advancements in kernel scheduling, with the full source code provided for reproducibility and further exploration.

### Gathering data:

1. Use VagrantFile to spawn up an Ubuntu VM.
2. Inside the VM, install nginx and apache-utils (it contains ab).
3. Start nginx
4. Start the ab tool to initialize load tests.
5. Start perf for a few seconds and record the data.
6. A perf.data file will be generated. Execute the command ``` sudo perf sched map > out_ab_nginx.txt ```.
7. Come out of the VM. Shut it down.
8. Create ```data``` directory in the root of this repo.
9. Move the out_ab_nginx.txt to the data folder.
10. Run the data_preprocessing.py to convert ```out_ab_nginx.txt``` to ```scheduling_data_out_ab_nging.csv```.
11. Run ```conda env create -f environment.yml``` to install conda environment.
12. Activate the created environment.
13. Run ```python train.py``` to train the model.
