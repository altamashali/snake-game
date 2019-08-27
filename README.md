# snakegame

This is a student project implementation of user Maurock's reinforcement learning snake game: https://github.com/maurock/snake-ga

This project was done for a class in which we use a cluster of raspberry pis to learn the basics of networking and parallel communications.

#### Modifications include:
  - Removal of graphs to save processing time and require installation of less libraries
  - Usage of mpirun to allow for multiple nodes to train concurrently
  - Modifications of saving and loading weights to allow the nodes to pull the best weights each time
      - We have the weights save with the score of that game included in the title of the file. So a score of 7 will save the weights file as "7weights.hdf5". The node which is choosing its weights to train off of will then pull the file with the highest score and train off of that.
  - Changes to displays (the first two nodes will show the game, the remainder will not)
  - Minor changes to the code to slightly optimize it, as well as changes to randomness values of movement in the early rounds of the games
  
  
#### To run:
To run our code dowload the directory and pip install:
- keras
- tensorflow
- pygame
- numpy
- pandas

Then open the directory snakegame and to run with four nodes as an example:

```python
mpirun -n 4 python3 snakev3.py
```




We must again thank Maurock for the initial code and our Professor, Dr. Tao for this fantastic learning opportunity.
