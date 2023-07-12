import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp

def csv_to_data(file_path, input_columns, output_columns, name_column):
    """ Loads data from a CSV file to be used into a DEA model.
    """
    
    df = pd.read_csv(file_path)
    
    X = np.array(df[input_columns])
    y = np.array(df[output_columns])
    names = list(df[name_column])
    
    return X, y, names

class DEA:
    
    def __init__(self, inputs, outputs, names):
        """
        Initialize the DEA object with input data.
            n = number of entities (observations)
            m = number of inputs (variables, features)
            r = number of outputs
        
        :param inputs: inputs, n x m numpy array
        :param outputs: outputs, n x r numpy array
        :return: self
        """
        
        # supplied data
        self.inputs = inputs
        self.outputs = outputs
        
        # parameters
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]
        
        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)
        
        # result arrays
        
        # output weights
        self.output_w = np.zeros((self.r, 1), dtype=float)
        
        # input weights
        self.input_w = np.zeros((self.m, 1), dtype=float)
        
        # unit efficiencies
        self.lambdas = np.zeros((self.n, 1), dtype=float)
        
        # thetas
        self.efficiency = np.zeros_like(self.lambdas)
        
        # names
        self.names = names
    
    def __efficiency(self, unit):
        """
        Efficiency function with already computed weights.
        
        :param unit: which unit to compute for
        :return: efficiency
        """
        
        # compute efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)
        
        return (numerator / denominator)[unit]
    
    def __target(self, x, unit):
        """
        Theta target function for one unit.
        
        :param x: combined weights
        :param unit: which production unit to compute
        :return: theta
        """
        
        # unroll the weights
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m + self.r)], x[(self.m + self.r):]
        
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)
        
        return numerator / denominator
    
    def __constraints(self, x, unit):
        """
        Constraints for optimization for one unit.
        
        :param x: combined weights
        :param unit: which production unit to compute
        :return: array of constraints
        """
        
        # unroll the weights
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m + self.r)], x[(self.m + self.r):]
        
        # init the constraint array
        constr = []
        
        # for each input, lambdas with inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t * self.inputs[unit, input] - lhs
            constr.append(cons)
        
        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)
        
        # for each unit
        for u in self.unit_:
            constr.append(lambdas[u])
        
        return np.array(constr)
    
    def __optimize(self):
        """
        Optimization of the DEA model.
        
        A = coefficients in the constraints
        b = rhs of constraints
        c = coefficients of the target function
        :return:
        """
        
        d0 = self.m + self.r + self.n
        
        # iterate over units
        for unit in self.unit_:
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,), iprint=0)
            
            # unroll weights
            self.input_w, self.output_w, self.lambdas = x0[:self.m], x0[self.m:(self.m+self.r)], x0[(self.m+self.r):]
            self.efficiency[unit] = self.__efficiency(unit)
    
    def fit(self):
        """
        Optimize the dataset, generate basic table.
        
        :return: table
        """
        
        # optimize
        self.__optimize()
    
    def summary(self):
        """
        Sumarize the results into a nice table.
        """
        
        print("Final thetas for each unit:\n")
        print("---------------------------\n")
        for n, eff in enumerate(self.efficiency):
            if len(self.names) > 0:
                name = "Unit %s" % self.names[n]
            else:
                name = "Unit %d" % (n+1)
            
            print("%s theta: %.4f" % (name, eff[0]))
            print("\n")
        
        print("---------------------------\n")
    
    def plot(self):
        """
        """
        
        pass

if __name__ == "__main__":
    X, y, names = csv_to_data("data/original_metjush_sample.csv", ["input1", "input2"], ["output1"], "unit")
    
    dea = DEA(X, y, names)
    
    dea.fit()
    dea.summary()
    dea.plot()
    