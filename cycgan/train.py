from solver import Solver
from param_parser import parameter_parser

if __name__ == '__main__':
    args = parameter_parser()
    solver = Solver(args)
    solver.train()