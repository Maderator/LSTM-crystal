More info about option to resume the bayesOpt read the section 
Object Functions in bayesOpt documentation.

## Hot to run experiment on Metacentrum
1. Use ssh to connect to Metacentrum.
2. Copy (or pull) whole source code (matlab folder) to Metacentrum.
3. Use flatten_files.py script to get all necessary files for given experiment to one folder
4. Go to the experiment folder
5. Use ```chmod +x *.sh``` for shell scripts to be executable
6. Optionally change the properties of the jobs being submitted in the shell scripts files
7. Use command ```./submit_jobs.sh``` for given experiment to submit prepared jobs
8. You can check output and error log files in the experiment folder on metacentrum
9. results are in results folder in the experiment folder 
10. use ```scp -r yourNickname@skirit.metacentrum.cz:~/LSTM-crystal-growth/metacentrum_experiments/experimentName/results matlab/experiments/experimentName``` in the LSTM-crystal-growth folder on your machine to copy results from the metacentrum after jobs successfuly end their execution

---
## Metacentrum experiments

We use theese toolboxes:
- Statistics and Machine Learning Toolbox (Statistics_Toolbox)
  - bayesopt
- Parallel Computing Toolbox (Distrib_Computing_Toolbox)
  - bayesopt when 'UseParallel' option is true
- Neural Network Toolbox (Neural_Network_Toolbox)
  - For netowrks definitions

### Scheduler
```Bash
qsub .. -l matlab=1 -l matlab_Statistics_Toolbox=1 -l Distrib_Computing_Toolbox=1 -l Neural_Network_Toolbox=1
```