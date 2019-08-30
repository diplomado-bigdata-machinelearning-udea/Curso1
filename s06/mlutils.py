from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
import sys
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def plot_2D_boundary(predict, mins, maxs, line_width=3, line_color="black", line_alpha=1, label=None):
    n = 200
    mins -= np.abs(mins)*.2
    maxs += np.abs(maxs)*.2
    d0 = np.linspace(mins[0], maxs[0],n)
    d1 = np.linspace(mins[1], maxs[1],n)
    gd0,gd1 = np.meshgrid(d0,d1)
    D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))
    preds = predict(D)
    levels = np.sort(np.unique(preds))
    levels = [np.min(levels)-1] + [np.mean(levels[i:i+2]) for i in range(len(levels)-1)] + [np.max(levels)+1]
    p = (preds*1.).reshape((n,n))
    plt.contour(gd0,gd1,p, levels=levels, alpha=line_alpha, colors=line_color, linewidths=line_width)
    if label is not None:
        plt.plot([0,0],[0,0], lw=line_width, color=line_color, label=label)
    return np.sum(p==0)*1./n**2, np.sum(p==1)*1./n**2


def plot_2Ddata_with_boundary(predict, X, y, line_width=3, line_alpha=1, line_color="black", dots_alpha=.5, label=None, noticks=False):
    mins,maxs = np.min(X,axis=0), np.max(X,axis=0)    
    plot_2Ddata(X,y,dots_alpha)
    p0, p1 = plot_2D_boundary(predict, mins, maxs, line_width, line_color, line_alpha, label )
    if noticks:
        plt.xticks([])
        plt.yticks([])
        
    return p0, p1



def plot_2Ddata(X, y, dots_alpha=.5, noticks=False):
    colors = cm.hsv(np.linspace(0, .7, len(np.unique(y))))
    for i, label in enumerate(np.unique(y)):
        plt.scatter(X[y==label][:,0], X[y==label][:,1], color=colors[i], alpha=dots_alpha)
    if noticks:
        plt.xticks([])
        plt.yticks([])


class Example_Bayes2DClassifier():
    
    def __init__ (self, mean0, cov0, mean1, cov1, w0=1, w1=1):
        self.rv0 = multivariate_normal(mean0, cov0)
        self.rv1 = multivariate_normal(mean1, cov1)
        self.w0  = w0
        self.w1  = w1

    def sample (self, n_samples=100):
        n = int(n_samples)
        n0 = int(n*1.*self.w0/(self.w0+self.w1))
        n1 = int(n) - n0
        X = np.vstack((self.rv0.rvs(n0), self.rv1.rvs(n1)))
        y = np.zeros(n)
        y[n0:] = 1
        
        return X,y
        
    def fit(self, X,y):
        pass
    
    def predict(self, X):
        p0 = self.rv0.pdf(X)
        p1 = self.rv1.pdf(X)
        return 1*(p1>p0)
    
    def score(self, X, y):
        return np.sum(self.predict(X)==y)*1./len(y)
    
    def analytic_score(self):
        """
        returns the analytic score on the knowledge of the probability distributions.
        the computation is a numeric approximation.
        """

        # first get limits for numeric computation. 
        # points all along the bounding box should have very low probability
        def get_boundingbox_probs(pdf, box_size):
            lp = np.linspace(-box_size,box_size,50)
            cp = np.ones(len(lp))*lp[0]
            bp = np.sum([pdf([x,y]) for x,y in zip(lp, cp)]  + \
                        [pdf([x,y]) for x,y in zip(lp, -cp)] + \
                        [pdf([y,x]) for x,y in zip(lp, cp)]  + \
                        [pdf([y,x]) for x,y in zip(lp, -cp)])
            return bp

        rngs = np.exp(np.arange(15))
        for rng in rngs:
            bp0 = get_boundingbox_probs(self.rv0.pdf, rng)
            bp1 = get_boundingbox_probs(self.rv1.pdf, rng)
            if bp0<1e-5 and bp1<1e-5:
                break

        if rng==rngs[-1]:
            print ("warning: bounding box prob size",rng,"has prob",np.max([bp0, bp1]))        
        
        
        # then, compute numerical approximation by building a grid
        mins, maxs = [-rng, -rng], [+rng, +rng]
        n = 100
        d0 = np.linspace(mins[0], maxs[0],n)
        d1 = np.linspace(mins[1], maxs[1],n)
        gd0,gd1 = np.meshgrid(d0,d1)
        D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))

        p0, p1 = self.rv0.pdf(D), self.rv1.pdf(D)

        # grid points where distrib 1 has greater probability than distrib 0
        gx = (p1-p0>0)*1.

        # true positive and true negative rates
        tnr = np.sum(p0*(1-gx))/np.sum(p0)
        tpr = np.sum(p1*gx)/np.sum(p1)
        return (self.w0*tnr+self.w1*tpr)/(self.w0+self.w1)  

def plot_estimator_border(bayes_classifier, estimators, n_samples=500):
    try:
        nns = [10,50,100]
        pbar = tqdm(total=len(estimators), ascii=False, file=sys.stdout, ncols=100)
        X,y = bayes_classifier.sample(n_samples)
        plt.figure(figsize=(15,3))
        for i,k in enumerate(sorted(estimators.keys())):
            estimator = estimators[k]
            pbar.update()
            plt.subplot(1,len(estimators),i+1)
            estimator.fit(X,y)
            plot_2Ddata_with_boundary(estimator.predict, X, y, line_width=2, dots_alpha=.3, label="estimator boundary")
            plot_2D_boundary(bayes_classifier.predict, np.min(X, axis=0), np.max(X, axis=0), 
                             line_width=4, line_alpha=.7, line_color="green", label="bayes boundary")
            plt.title(k+", estimator=%.3f"%estimator.score(X,y)+ "\nanalytic=%.3f"%bayes_classifier.analytic_score())
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pbar.close()
    except BaseException as e:
        pbar.close()
        raise e

def sample_borders(mc, estimator, samples, n_reps):
    try:
        pbar = tqdm(total=len(samples)*n_reps, ascii=False, file=sys.stdout, ncols=100)
        plt.figure(figsize=(15,3))
        for i,n_samples in enumerate(samples):
            plt.subplot(1,len(samples),i+1)
            for ii in range(n_reps):
                pbar.update()
                X,y = mc.sample(n_samples)
                estimator.fit(X,y)
                if ii==0:
                    plot_2D_boundary(estimator.predict, np.min(X, axis=0), np.max(X, axis=0), 
                                     line_width=1, line_alpha=.5, label="estimator boundaries")
                else:
                    plot_2D_boundary(estimator.predict, np.min(X, axis=0), np.max(X, axis=0), 
                                     line_width=1, line_alpha=.5)                    
                plt.title("n samples="+str(n_samples))
            plot_2D_boundary(mc.predict, np.min(X, axis=0), np.max(X, axis=0), 
                             line_width=5, line_alpha=1., line_color="green", label="bayes boundary")
        pbar.close()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    except BaseException as e:
        pbar.close()
        raise e

from sklearn.neighbors import KernelDensity

class KDClassifier:
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, X,y):
        """
        builds a kernel density estimator for each class
        """
        self.kdes = {}
        for c in np.unique(y):
            self.kdes[c] = KernelDensity(**self.kwargs)
            self.kdes[c].fit(X[y==c])
        return self
        
    def predict(self, X):
        """
        predicts the class with highest kernel density probability
        """
        classes = self.kdes.keys()
        preds = []
        for i in sorted(classes):
            preds.append(self.kdes[i].score_samples(X))
        preds = np.array(preds).T
        preds = preds.argmax(axis=1)
        preds = np.array([classes[i] for i in preds]) 
        return preds
    
    def score(self, X, y):
    
        return np.mean(y==self.predict(X))
    
    
def accuracy(y,preds):
    return np.mean(y==preds)

    
from sklearn.model_selection import train_test_split
def bootstrapcv(estimator, X,y, test_size, n_reps, score_func=None, score_funcs=None):

    if score_funcs is None and score_func is None:
        raise ValueError("must set score_func or score_funcs")
    
    if score_funcs is not None and score_func is not None:
        raise ValueError("cannot set both score_func and score_funcs")
    
    if score_func is not None:
        rtr, rts = [],[]
    else:
        rtr = {i.__name__:[] for i in score_funcs}
        rts = {i.__name__:[] for i in score_funcs}
        
    for i in range(n_reps):
        Xtr, Xts, ytr, yts = train_test_split(X,y,test_size=test_size)
        estimator.fit(Xtr, ytr)
        if score_func is not None:
            rts.append(score_func(yts, estimator.predict(Xts)))
            rtr.append(score_func(ytr, estimator.predict(Xtr)))
        else:
            for f in score_funcs:
                fname =  f.__name__
                rts[fname].append(f(yts, estimator.predict(Xts)))
                rtr[fname].append(f(ytr, estimator.predict(Xtr)))
    if score_func is not None:
        return np.array(rtr), np.array(rts)
    else:
        rtr = {i: np.array(rtr[i]) for i in rtr.keys()}
        rts = {i: np.array(rts[i]) for i in rts.keys()}
        return rtr, rts

def lcurve(estimator, X,y, n_reps, score_func, show_progress=False):
    test_sizes = np.linspace(.9,.1,9)
    trmeans, trstds, tsmeans, tsstds = [], [], [], []
    if show_progress:
        pbar = tqdm(total=len(test_sizes), ascii=False, desc="lcurve "+estimator.__class__.__name__, file=sys.stdout, ncols=100, unit=" train/test split")
    try:
        for test_size in test_sizes:
            rtr, rts = bootstrapcv(estimator,X,y,test_size,n_reps, score_func)
            trmeans.append(np.mean(rtr))
            trstds.append(np.std(rtr))
            tsmeans.append(np.mean(rts))
            tsstds.append(np.std(rts))
            if show_progress:
                pbar.update(1)
    except (Exception, KeyboardInterrupt) as e:
        if show_progress:
            pbar.close()
        raise e       
    trmeans = np.array(trmeans)
    trstds  = np.array(trstds)
    tsmeans = np.array(tsmeans)
    trstds  = np.array(tsstds)
    abs_train_sizes = len(X)*(1-test_sizes)
    plt.plot(abs_train_sizes, trmeans, marker="o", color="red", label="train")
    plt.fill_between(abs_train_sizes, trmeans-trstds, trmeans+trstds, color="red", alpha=.2)
    plt.plot(abs_train_sizes, tsmeans, marker="o", color="green", label="test")
    plt.fill_between(abs_train_sizes, tsmeans-tsstds, tsmeans+tsstds, color="green", alpha=.2)
    plt.xlim(len(X)*.05, len(X)*.95)
    plt.xticks(abs_train_sizes)
    plt.grid()
    plt.xlabel("train size")
    plt.ylabel(score_func.__name__)
    plt.ylim(0,1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
              ncol=2, fancybox=True, shadow=True)
