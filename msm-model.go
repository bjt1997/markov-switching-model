package main

import (
	"fmt"
	"math"
	"os"
	"sort"

	"math/bits"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"

	"time"

	"gonum.org/v1/gonum/stat"

	"gonum.org/v1/gonum/stat/distuv"

	"gonum.org/v1/gonum/optimize"
)

const ISPI = 1.0 / math.Sqrt2 / math.SqrtPi

// Compute transition probability matrix
func probMat(p []float64) *mat.Dense {
	x := 1
	a := mat.NewDense(2, 2, []float64{1, 1, 1, 1})
	c := mat.NewDense(1, 1, []float64{1.0})
	var tmp mat.Dense
	for i := range p {
		gi := p[i] * 0.5
		a.Set(0, 0, 1.0-gi)
		a.Set(0, 1, gi)
		a.Set(1, 0, gi)
		a.Set(1, 1, 1.0-gi)
		tmp.Kronecker(c, a)
		x = x * 2
		c.Grow(x, x)
		c.CloneFrom(&tmp)
		tmp.Reset()
	}
	return c
}

// Compute volatilities corresponding to states
func sigma(m0 float64, s0 float64, k int, M []int) []float64 {
	n := len(M)
	s := make([]float64, n)
	m1 := 2.0 - m0
	for i := range s {
		s[i] = s0 * math.Sqrt(math.Pow(m1, float64(M[i]))*math.Pow(m0, float64(k-M[i])))
	}
	return s
}

/* States expressed as number of "ON" states in total as that is what matters in computing volatility
 */
func states(k int) []int {
	n := int(math.Pow(2.0, float64(k)))
	st := make([]int, n)
	for i := range st {
		st[i] = bits.OnesCount(uint(i))
	}
	return st
}

// Sigmoid function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*x))
}

// Compute negative log likelihood of MSM model
func negloglik(par []float64, x []float64, k int, states []int) float64 {
	// Transform (-inf, inf) domain to appropriate parameter domains
	p := transformParams(par)

	n := len(states)

	// Compute transition probability matrix for product of Markov chains
	A := probMat(p[2:])

	// Compute volatility of each state
	s := sigma(p[0], p[1], k, states)

	// Initialise unconditional probability distribution of states
	B := make([]float64, n)
	for i := range B {
		B[i] = 1.0 / float64(n)
	}

	// Auxiliary variables
	wx := make([]float64, n)
	var tmp mat.Dense
	sw := 0.0

	// Initialise log-likelihood
	ll := 0.0

	for i := range x {
		for j := range wx {
			wx[j] = ISPI * math.Exp(-0.5*x[i]*x[i]/s[j]/s[j]) / s[j]
		}
		tmp.Mul(A, mat.NewDense(n, 1, B))
		sw = 0.0
		for j := range wx {
			sw += wx[j] * tmp.At(j, 0)
		}
		ll += math.Log(sw)
		for j := range B {
			B[j] = wx[j] * tmp.At(j, 0) / sw
		}
	}
	return -ll
}

func transformParams(par []float64) []float64 {
	m := len(par)
	p := make([]float64, m)
	// Transform (-inf, inf) domain to (1,2) and (0,inf)
	p[0], p[1] = 1.0+sigmoid(par[0]), math.Exp(par[1])

	// Transform (-inf, inf) domain to probabilities
	for i := 2; i < m; i++ {
		p[i] = sigmoid(par[i])
	}
	return p
}

func simulate(par []float64, nsims int) []float64 {
	m0, s0 := par[0], par[1]
	p := par[2:]
	k := len(p)
	A := probMat(p)
	M := states(k)
	s := sigma(m0, s0, k, M)
	x := make([]float64, nsims)
	var st int
	q := mat.Row(nil, 0, A)
	d2 := distuv.NewCategorical(q, rand.NewSource(uint64(time.Now().UnixNano())))
	st = int(d2.Rand())
	d1 := distuv.Normal{Mu: 0.0, Sigma: 1.0}
	x[0] = d1.Rand() * s[st]

	for i := 1; i < nsims; i++ {
		q = mat.Row(nil, st, A)
		d2 = distuv.NewCategorical(q, rand.NewSource(uint64(time.Now().UnixNano())))
		st = int(d2.Rand())
		x[i] = d1.Rand() * s[st]
	}
	return x
}

func predict(par []float64, paths int, pathLen int) []float64 {
	vol := make([]float64, paths)
	ret := make([]float64, pathLen)
	for i := range vol {
		ret = simulate(par, pathLen)
		vol[i] = stat.StdDev(ret, nil)
	}
	res := make([]float64, 2)
	res[0], res[1] = stat.MeanStdDev(vol, nil)
	fmt.Printf("-------------- Vol prediction ---------------\n")
	fmt.Printf("Estimate:\t%0.4f\n", res[0])
	fmt.Printf("Std Error:\t%0.4f\n", res[1]/math.Sqrt(float64(paths)))
	fmt.Printf("----------------------------------------------\n")
	return res
}

// Fit an MSM-BT model of dimension k to data x
func fit(x []float64, k int) []float64 {
	// initialise parameters for negloglik function
	par := make([]float64, k+2)
	dist := distuv.Normal{Mu: 0.0, Sigma: 1.0}
	for i := range par {
		par[i] = dist.Rand()
	}
	// Use sample standard deviation for s0 param of model
	sd := stat.StdDev(x, nil)
	par[1] = math.Log(sd)

	// calculate Markov chain states
	M := states(k)

	// compute negative of log likelihood
	start := time.Now()
	problem := optimize.Problem{
		Func: func(par []float64) float64 {
			return negloglik(par, x, k, M)
		},
	}

	result, err := optimize.Minimize(problem, par, nil, &optimize.NelderMead{})
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	fmt.Printf("---------- MSM-BT Model Fit Results ----------\n")
	fmt.Printf("MLE took %v seconds\n", time.Since(start))
	fmt.Printf("Numberof func evals: %d\n", result.Stats.FuncEvaluations)
	fmt.Printf("Status:\t%v\n", result.Status)
	fmt.Printf("Loglik:\t%0.f\n", result.F)
	res := transformParams(result.X)
	fmt.Printf("-------------- Model parameters --------------\n")
	fmt.Printf("m0:\t%0.4f\n", res[0])
	fmt.Printf("s0:\t%0.4f\n", res[1])
	ps := res[2:]
	sort.Float64s(ps)
	for i := range ps {
		fmt.Printf("p%d:\t%0.4f\n", i+1, ps[i])
	}
	// Return model parameters and objective value
	res = append(res, result.F)

	return res
}
