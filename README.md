# Markov Switching Multi-Fractal model

## The MSM Model
De-meaned returns are modelled as r_t = s_t N(0, 1), where r_t is return at time t and s_t is (heteroskedastic) volatility modelled as s_t = s0 sqrt(M_1 M_2 ... M_k). k is the model dimension and M_i are 2-state Markov Chains.

Refer to Calvet and Fisher (2001)[^1]

[^1]: Calvet, L.; Fisher, A. (2001). "Forecasting multifractal volatility" [(PDF)](https://archive.nyu.edu/bitstream/2451/26894/2/wpa99017.pdf). Journal of Econometrics. 105: 27–58.

## My Variation aka BT Variation
The original model parametrises the transition probabilities of Markov chains M_i with just two parameters. So the model is parsimonious and parameters do not grow with k. In BT variation, each chain M_i is characterised by a probability parameter g_i (transition matrix [1-gi gi; gi 1-gi]). This allows k-2 more degrees of freedom.

## Go Implementation
The model is implemented as unconstrained optimisation with suitable parameter transformation. There are a total of k+2 parameters, m0 the binomial pdf paramter (1< m0 <2), s0 (> 0) the unconditional volatility and k probabilities. m0 is mapped into (-inf, inf) with inverse sigmoid, s0 with log and probabilities with inverse sigmoid.

## Usage
```
msm -k 4 -n 200 -w 240 -i input.csv -o results.csv
```

Command line flags:

```
 -i string
    	Name of csv file containing return data
  -k int
    	MSM model dimension, +ve integer, caution: k > 10 can take minutes to run
  -n int
    	Number of simulations for computing mean and std dev of vol estimate, minimum 100 (default 200)
  -o string
    	Name of output file (default "results.csv")
  -w int
    	Time window for vol prediction in number of bars, minimum 30 (default 30)
```

The output file is a csv containing model parameters, negative log likelihood, model dimension, mean of predicted vol and std dev of predicted vol. Std error of mean can be inferred as std dev / sqrt(n).
