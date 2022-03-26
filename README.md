### About:
In the modern Linux kernel, the scheduling algorithm is responsible for deciding which process gets executed. The current default scheduler in the Linux kernel, the Completely Fair Scheduler (CFS), aims to maximize the overall CPU utilization while also maximizing interactive performance. In this project, we attempt to learn the behaviour of CFS using recent machine learning techniques. The intuition behind doing this is to evaluate the possibility of building a generalized task scheduler for any kind of workload. The important contributions of this project are: 1) Extracting and building a CFS scheduling dataset from a running Linux kernel. 2) Training and evaluating a deep learning model on this obtained dataset.

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
