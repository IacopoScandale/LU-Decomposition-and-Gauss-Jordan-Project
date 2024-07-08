# Matlab Class Project

A MATLAB project on LU decomposition and its applications, enhanced with the Gauss-Jordan algorithm. This project was created for the Numerical Methods course and presented during an oral exam at Sapienza University of Rome.



## Brief Project Explaination
This MATLAB project focuses on LU decomposition and its applications, extended with the Gauss-Jordan algorithm. It covers:

* **LU Decomposition with and without Pivoting:**
    Implements `my_lu` and `my_lu_pivoting` to decompose a matrix A into L, U, and P.
    *Applications*:
    * Solve linear systems Ax=b using the decomposed matrices. 
    * Calculate Matrix Determinant `my_det`
<br>
* **Running Time Analysis:**
    Compares the performance of the custom LU decomposition (my_lu_pivoting) with MATLAB’s built-in `lu` function. Same for all other created functions.
<br>
* **Test Matrices:**
    * Random Matrices
    * Poisson Matrices
    * Hilbert Matrices
    * Singular Matrices
<br>
* **Gauss-Jordan Algorithm:**
    Implements `my_gauss_jordan` for solving linear systems and computing reduced row echelon form (RREF).
    Includes applications for matrix inversion.
<br>
* **Fill-in Visualization:**
    Plots sparsity patterns of matrices before and after transformations using MATLAB’s spy function.



## Saved Time Arrays
Due to time constraints for the oral exam, I saved all the timing vectors by running the code on my computer. For this reason some code blocks are commented (in italian because this course was taught in italian).

If you want you can use saved times, or uncomment blocks and re-run the project.