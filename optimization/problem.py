# -*- coding: utf-8 -*-

r"""
High level description of the module.
"""

# TODO:
#   * Fix docstrings.
#   * Re-factor to remove repetition.

import sys, os
import importlib

import predictive_functions
importlib.reload(predictive_functions)
import objective_functions
importlib.reload(objective_functions)
import optimization_algs
importlib.reload(optimization_algs)
#import matplotlib.pyplot as plt
import data_manipulation as dm
importlib.reload(dm)
import numpy as np
import early_detection
importlib.reload(early_detection)
#import matplotlib.cm as cm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


class FS_problem():
    #  def __init__(self, goal, data, objective, threshold=2, log_FPR_threshold=0,
            #  predictor='lin_reg', num_samples=100000, dl=0.1, tol=0.5, maxiter=10,
            #  n_folds=1, bootstrap=False, early_detection = False):
    def __init__(self, goal, data, req_data, objective, threshold=2,
            log_ATFS_threshold=0, predictor='lin_reg', num_samples=100000,
            dl=0.1, tol=0.5, maxiter=10, n_folds=1, bootstrap=False,
            early_detection = False,OOS_data=None,lin_reg_intercept=False):
        r"""Class for the feature selection optimization problem

        Provides an easy way to try several existing feature selection
        algorithms and add more

        Parameters
        ----------
        goal : dictionary
            goal data
        data : list
            data sources
        req_data : list
            data sources that must be included in optimization.
        objective : string
            specifies the type of objective function to use in optimization
        predictor : string
            the type of predictive function to use in optimization
        n_folds :
            number of folds for cross-validation
            (n_folds=1 for in-sample testing)
        bootstrap : boolean
            True if training using bootstrap sample
        OOS_data: list containing gold standard and data sources for OOS dates
        lin_reg_intercept: whether to include an intercept when doing linear
            regressions

        References
        ---------
        Optimizing Provider Recruitment for Influenza Surveillance Networks

        """
        self.goal, self.data, self.req_data = goal, data, req_data
        self.objective = objective
        self.n_folds = n_folds
        self.bootstrap = bootstrap
        self.early_detection = early_detection
        if self.early_detection:
            self.threshold = threshold
            #  self.log_FPR_threshold = log_FPR_threshold
            self.log_ATFS_threshold = log_ATFS_threshold
            self.num_samples = num_samples
            self.dl = dl
            self.tol = tol
            self.maxiter = maxiter
        else:
            self.predictor = predictor
        # initialize values to be computed
        self.optimal_value = 0
        self.optimum = []
        self.objective_trace = []
        if OOS_data is not None:
            self.OOS_goal, self.OOS_data = OOS_data
        else:
            self.OOS_goal, self.OOS_data = None,None
        self.lin_reg_intercept = lin_reg_intercept
        

    def predictive_function(self, goal_series, series_list):
        r"""Returns a predictive time series

        This function takes goal along with the time
        series in series_list to create a time series that fits
        goal_series using the time series in series_list as best as
        possible.

        Parameters
        ----------
        goal_series : list
            time series used as goal
        series_list : list
            list of time series for fitting

        Notes
        -----
        self.predictor much match the name of a function in
        predictive_functions.py

        Returns
        -------
        return : list
            time series which best replicates goal_series

        """
        if hasattr(predictive_functions, self.predictor):
            return eval("predictive_functions." + self.
                        predictor)(goal_series, series_list)
        else:
            print(self.predictor + " is not an predictive function")
            sys.exit()

    def objective_function(self, subset, **kwargs):
        r"""Evaluates objective function on specified subset. Returns distance
        between self.goal and the series in subset

        Notes
        -----
        self.objective must match the name of a function in
        objective_functions.py

        Parameters
        ----------
        subset : list
            subset of data dictionaries from data

        Returns
        -------
        return : float
            a measure of distance

        """
        subset = dm.listify(subset)
        if hasattr(objective_functions, self.objective):
            return eval("objective_functions." + self.
                        objective)(self, subset, **kwargs)
        elif hasattr(early_detection, self.objective):
            return eval("early_detection." + self.
                        objective)(self, subset, **kwargs)
        else:
            print(self.objective + " is not an objective function")
            sys.exit()
    
    def test_OOS(self, subset, **kwargs):
        r"""Evaluates objective function on specified subset. Returns distance
        between self.goal and the series in subset

        Notes
        -----
        

        Parameters
        ----------
        subset : list
            subset of data dictionaries from data

        Returns
        -------
        return : float
            a measure of distance

        """
        subset = dm.listify(subset)
        return objective_functions.R_squared_OOS(self, subset, **kwargs)
        
        

    def graph_data_sources(self, sources):
        r"""Plots the data sources from sources and prints a legend

        Parameters
        ----------
        sources : list
            subset of data dictionaries from data

        """
        sources = dm.listify(sources)
        for source in sources:
            source = dm.normalize(source)
            time = source['data']['times']
            values = source['data']['values']
            name = source['metadata']['name']
            subname = source['metadata']['subname']
            plt.plot(time, values, label = name + " | " + subname)
        plt.legend(loc=2, prop={'size': 10})
        plt.show()


    def graph_results_earlydetection(self):
        r"""Plots events generated by goal along with predictive function of
        probability of event occuring

        """
        optimum_series_list = [datum['data']['values'] for datum in self.
                               get_optimum()]
        goal_series = self.goal['data']['values']
        coefficients, predictive = self.predictive_function(goal_series,
                                                 optimum_series_list)
        #time = range(len(optimum_series_list[0]))
        time = self.goal['data']['times']
        plt.figure()
        plt.plot(time, goal_series, label='Gold standard')
        plt.plot(time, predictive, label='Predictive')
        plt.legend(loc=2, prop={'size': 10})
        # x1,x2,y1,y2 = plt.axis()
        # plt.imshow([predictive], extent=(x1,x2,y1,y2),
        #         cmap=cm.GnBu, alpha=0.5) # or use cm.RdPu?
        plt.show()

    def graph_results_series(self):
        r"""Plots predictive function for self.optimum along with goal

        """
        optimum_series_list = [datum['data']['values'] for datum in self.
                               get_optimum()]
        goal_series = self.goal['data']['values']
        coefficients, predictive = self.predictive_function(goal_series,
                                                 optimum_series_list)
        #time = range(len(optimum_series_list[0]))
        time = self.goal['data']['times']
        plt.figure()
        plt.plot(time, goal_series, label='Gold standard')
        plt.plot(time, predictive, label='Predictive')
        plt.legend(loc=2, prop={'size': 10})
        # x1,x2,y1,y2 = plt.axis()
        # plt.imshow([predictive], extent=(x1,x2,y1,y2),
        #         cmap=cm.GnBu, alpha=0.5) # or use cm.RdPu?
        plt.show()

    def graph_results_scatter(self):
        # TODO:
        # - Fix plotting so new figure is not produced upon every call of
        #   method (graph_results_series seems to only produce one figure and
        #   write over it, why?)
        r"""Plots scatterplot of predictive function for self.optimum versus
        goal

        """
        optimum_series_list = [datum['data']['values'] for datum in self.
                               get_optimum()]
        goal_series = self.goal['data']['values']
        coefficients, predictive = self.predictive_function(goal_series,
                                                 optimum_series_list)
        _, ax = plt.subplots()
        ax.scatter(goal_series, predictive, s=2, c='purple', alpha=0.8)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]
        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel('Gold standard values')
        plt.ylabel('Predictive values')
        plt.show()

    def graph_results_ranked(self):
        r"""Plots performance curve of cumulative set of ranked data sources

        """
        ranked_set = []
        for datum in self.optimum:
            ranked_set.append(datum['metadata']['name'] + " | " +
                    datum['metadata']['subname'])
        plt.figure()
        plt.bar(range(len(ranked_set)), self.objective_trace,
                align='center')
        plt.xticks(range(len(ranked_set)), ranked_set,
                rotation='vertical', fontsize=10)
        # plt.xlabel('Cumulative dataset')
        plt.ylabel(self.objective)
        plt.show()
        plt.subplots_adjust(bottom=0.5)

    def print_solution(self):
        r"""Prints the current optimal solution

        """
        print("Optimal solution: ")
        #for datum in self.optimum:
        #    print datum['metadata']['name']
        dm.info(self.optimum)
        print("\nObjective value: " + str(self.optimal_value) + '\n')
        if self.objective_trace:
            print("Objective trace: ")
            for value in self.objective_trace:
                print(str(value))

    def optimize(self, algorithm="forward_selection", **kwargs):
        r"""Master function for calling the optimization routine

        Notes
        -----
        The keyword argument algorithm must match the name of a function
        in optimization_algs.py

        Parameters
        ----------
        algorithm : string
            specifies what algorithm to use

        """
        if hasattr(optimization_algs, algorithm):
            solution = eval("optimization_algs." + algorithm)(self, **kwargs)
            self.optimum = solution[0]
            self.optimal_value = solution[1]
            self.objective_trace = solution[2]
            self.objective_values_single_datum = solution[3]
            self.OOS_R_squared = solution[4]
        else:
            print(algorithm + " is not an optimization algorithm")
            sys.exit()
        #self.print_solution()
        # self.graph_results_series()
        #if algorithm == "forward_selection":
            #self.graph_results_ranked()
        #self.graph_results_scatter()
        return self.optimum, self.objective_trace, self.objective_values_single_datum,\
            self.OOS_R_squared


    def get_objective_value(self):
        r"""Returns optimal_value

        Returns
        -------
        self.optimal_value : type
            Description

        """
        return self.optimal_value

    def get_optimum(self):
        r"""Returns optimum

        Returns
        -------
        self.optimum : type
            Description

        """
        return self.optimum

    def get_objective_trace(self):
        r"""Returns objective trace

        Returns
        -------
        self.objective_traces : type
            Description

        """
        return self.objective_trace
