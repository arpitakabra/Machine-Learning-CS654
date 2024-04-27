# ES654-2020 Assignment 3

*Arpita Kabra | Ayush Anand* - *18110026 | 19110074*

------

> Write the answers for the subjective questions here

The multicollinear dataset is made by adding another feature column which is summation of two existing columns in the data. The obtained error on the predicted values are not different as compared to the non-collinear data. However, the interpretability of results change based on the extent of collinearity.

For instance, theta values do not shoot up when only two columns are added and a third column is generated. However, if the added column is scaled version of a previous column, theta shoots up, and so does the error. The model is no longer able to fit the data.