# Statistical Testing

- see visualization of experiment for example of usage.

## Statistical Comparison of Multiple Algorithms
The comparison of multiple algorithms over (multiple) data sets can be done by many methods. Statistical test is only one of possible methods. Demšar in his 2006 paper "Statistical Comparisons of Classiﬁers over Multiple Data Sets" recommends **Friedman test** as the go to approach for null hypothesis testing.

### Friedman Test
We use this python function:
```python
scipy.stats.friedmanchisquare(*samples)
``` 
which returns:
- statistics: The test statistic, correcting for ties.
- pvalue: The associated p-value assuming that the test statistic has a chi squared distribution.

TODO: Does MSE (RMSE) have chi-square distribution? 

## Post-hoc analysis
The mean-ranks post-hoc test is commonly used in machine learning however the comparison between two algorithms A and B depends also on the performance of other (m-2) algorithms which leads to some paradoxical situations.

Therefore, Bernavoli et al. in 2006 paper "Should We Really Use Post-Hoc Tests Based on Mean-Ranks?" recommend to perform the pairwise compariosn of the post-hoc analysis using the **Wilcoxon signed-rank test** or the **signed test**.

We use the **Wilcoxon signed-rank test** python function:
```python
scipy.stats.wilcoxon(x, y=None)
```
TODO: check the wilcoxon parameters and add them to the function example above
TODO: check for the bayesian alternatives (scipy has monte carlo methods but benavoli et al. uses Dirichlet process)

We then order the corresponding statistics and *p*-values. The p-value ($\alpha$) is then adjusted by procedures like Bonferroni and Nemenyi which are one-step procedures, step-up procedures like Hochberg and Hommel, and step-down procedures like Holm, Nemenyi, Shaffer, Bergmann-Hommel. Garcia end Herrera in "An Extension on “Statistical Comparisons of Classifiers over Multiple Data Sets” for all Pairwise Comparisons" do not recommend use of Nemenyi, but recommend Holm, Shaffer and Bergmann-Hommel. Bergmann-Hommel is the best performing one but the most difficult one. Shaffer is strongly encouraged with respect to Holm procedure. Holm is the easiest one followed by Shaffer. We use this python method for *p*-value adjustment:
```python
statsmodels.stats.multitest.multipletests(p_values, method=..., alpha=0.05)
```
Where for each procedure we change the method parameter to:
- bonferroni: "bonferroni"
- holm: "holm"
- shaffer: not implemented anywhere
- Bergmann-Hommel: not implemented anywhere

## Critical Difference Diagram
- see visualization of experiment for example of usage.
- For diagram visualization of statistiacal test results, we use critdd Diagram class to create the critical difference diagrams.
- Example of code:
```python
from critdd import Diagram

diagram = Diagram(x,
    treatment_names=treatment_names,
    maximize_outcome = False,
)

diagram.to_file(
    "critdd.tex",
    alpha=0.05,
    adjustment="holm",
    reverse_x=True,
)
```
where x is the matrix of the experiment samples for each treatment and treatment_names is the list of the names of the treatments.



