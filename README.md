# Proximal-bundle-method

A python implementation of proximal bundle method for non-smooth unconstrained convex optimization.

## Description

This product can solve non-smooth unconstrained convex optimization problems without commercial solvers.

## Demo

![Demo](https://github.com/kohei-harada/proximal-bundle-method/usage.gif)

## Requirement

- scipy
- numpy
- autograd
- cvxopt

## Usage

Just kick
```
python bundle.py
```
It begins to solve problem/sample.py.
You can set the target problem on "testset" array at problems.py.
If you want to solve your original problem, create a new .py file(or copying sample.py) at problems directory and edit.
Necessary requirements for a problem file is written in sample.py.

## Installation

```
git clone https://github.com/kohei-harada/proximal-bundle-method
```

## Author

Kohei Harada

## License

MIT
