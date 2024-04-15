**This is a tutorial which should introdue future master students at INI into the topic of SNNs
while also showing them how to implement and train them using JAX.**

### Authors: Tristan Torchet and Elija Vida

# Chapter 0: Git Introduction

https://www.youtube.com/watch?v=Z9fIBT2NBGY

-> Only if you are not accustomed to using Git, otherwise this chapter can be skipped.

## Assignment: 
Watch the video and try to understand how to use git, 
summarize all take-aways including the most common commands in git.


# Chapter 1: Literature Review 

Reading List to get familiarized with SNNs:

1. Mosaic paper
2. "The neuromorphic Bible"
3. A deep learning framework for neuroscience


# Chapter 2: JAX tutorial

Tutorial 1: https://youtu.be/SstuvS-tVc0?feature=shared
Tutorial 2: https://www.youtube.com/watch?v=CQQaifxuFcs
Tutorial 3: https://www.youtube.com/watch?v=6_PqUPxRmjY

## Assignment: 
Go through all three tutorials, create your own notebooks for each individual tutorial 
and try to write down your own notes on parts that seem interesting or new to you.

# Chapter 3: SNNtorch tutorial

## Advice: Read the bible before!

Tutorial 1: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html
Tutorial 2: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html
Tutorial 3: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html

## Assignment: 
The main goal is to implement a LIF-neuron model in JAX. Go through all three tutorials, 
create a notebook for each, replicate the code and play around with it while at the same time 
try to understand the funadmental concepts of SNNs. 
While understanding the concepts, try to keep in mind to implement a LIF-neuron model in JAX 
using your aquired knowledge from the JAX tutorials.

-> For this, there are two sub-assignments that might be helpful:
1. In the JAX-documentation, try to understand the functioning of the different loop-functions of JAX,
namely "jax.lax.fori_loop(), jax.lax.scan(), and jax.lax.while_loop()".
2. In the python documentation, try to understand the different timing functions, 
namely "time" and "timeit", and write down their main differences.

# Chapter 4: MNIST Classification

## Assignment: 
Go through the notebook chronologically and solve the TODOs. 
If you are stuck, you can look at the solution as a reference.

--> IMORTANT: it should be forbidden to use Copilot for this tutorial, 
most of the understanding stems from coding the functions yourself!

References: Stochastic Gradient Descent 
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiQl93siLiEAxXr6wIHHYRwC7cQFnoECBgQAQ&url=https%3A%2F%2Ftowardsdatascience.com%2Fbatch-mini-batch-stochastic-gradient-descent-7a62ecba642a&usg=AOvVaw0RbBf_HM7eb1HNO-PYXoTT&opi=89978449
https://kenndanielso.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_6_Stochastic_and_minibatch_gradient_descent.html


# Chapter 5: ECG Classification 

## Assignment: 
Go through the notebook chronologically and solve the TODOs. 
If you are stuck, you can look at the solution as a reference.

--> IMORTANT: it should be forbidden to use Copilot for this tutorial, 
most of the understanding stems from coding the functions yourself!