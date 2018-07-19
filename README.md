
# smarties

This repository provides ostensibly high-performance c++ implementations of [RACER](https://arxiv.org/abs/1807.05827) and other popular RL algorithms including PPO, DQN, DPG, ACER, and NAF.  
Smarties was largely written during the development of [ReF-ER and RACER](https://arxiv.org/abs/1807.05827), and was an honest attempt to maximize code reusability, modularity, and extensibility.  
Development should be considered an on-going process, however RACER and DPG were extensively tested and analyzed on the mujoco-based robotic benchmark problems of OpenAI gym and the DeepMind Control Suite.  

Smarties requires gcc version 6.1 or greater, a thread-safe (at least `MPI_THREAD_SERIALIZED`) implementation of MPI, and a serial BLAS implementation with CBLAS interface. Furthermore, in order to test on the benchmark problems, OpenAI gym or the DeepMind Control Suite with python>=3.5.  

The `makefiles` folder is poorly documented and requires some experience with gnu makefiles to navigate. However, compilation on Mac OS should work out of the box (assuming all dependencies are installed).  

It should be quite easy for any user to extend this repository by adding their own environment definitions written either with python or c++. Some examples are provided in the `apps` folder.  

The folder `launch` contains the launch scripts, some description on how to use them, and the description of the output files. Some tools to postprocess the outputs are in the folder `pytools`.  

To cite this repository, reference the paper:
```
@article{novati2018a,
    title={Remember and Forget for Experience Replay},
    author={Novati, G and Koumoutsakos, P},
    journal={arXiv preprint arXiv:1807.05827},
    year={2018}
}
```
