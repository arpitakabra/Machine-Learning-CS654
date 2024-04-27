# ES654-2020 Assignment 3

*Arpita Kabra | Ayush Anand* - *18110026 | 19110074*

------

> Write the answers for the subjective questions here

The following plots show the timing analysis comparing normal method and gradient descent with varying sample and feature sizes.

Varying Sample Sizes:
![alt text](https://github.com/ES654/assignment-2-arpita-ayush/blob/master/assignment-2/q8_varying_N.png)

Varying Feature Size
![alt text](https://github.com/ES654/assignment-2-arpita-ayush/blob/master/assignment-2/q8_varying_D.png)

In both the cases, for gradient descent, batch size is kept equal to the number of training samples (Vanilla Gradient Descent). We observe results similar to theoritical analysis.
Time complexity of gradient descent is O(ND^2)  and for Normal Method it is O(D^3). However, time complexity of gradient descent also depends on the number of iterations. Hence, if number of features are not very learge, normal method is a better approach to use. 
