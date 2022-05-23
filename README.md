# Custom_RL_Environments
Custom RL Envs which implement the gym interface to test RL algorithms on


Environments provided:
1. Traffic Signal env: simulates traffic flow at a stop signal using a 1 D burgers equation (PDE)
   control variable is the timing of the stop sign (red to green) for two roads-NS and EW meeting
   at the junction.stop time has to be between a min_time and max_time value
   
2. Custom Pong env: is a wrapper around atari's Pong env, where tabular features are derived from
   stacked game frames (images). The features used are: 
   a) paddle positions (player and opponent)
   b) puck velocity X and Y
   c) puck position X and Y
   
   
3. LQR env: an RL env representing the discrete LQR control problem with continuous control input. 
   The objective is to match the analytical max reward/min cost calculated by the optimal LQR controller
