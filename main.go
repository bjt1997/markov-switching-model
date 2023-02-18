/*
Markov Switching Multi-fractal (MSM) model for volatility prediction.
This is a BT variant implementation, where the transition probabiltiies of Markov chains driving the volatility process are modelled directly.
*/

package main

import (
	"encoding/csv"

	"fmt"

	"strconv"

	"os"

	"gonum.org/v1/gonum/stat"

	"flag"
)

var (
	msmdim, window, samples int
	infile, outfile         string
)

func init() {
	flag.IntVar(&msmdim, "k", 0, "MSM model dimension, +ve integer, caution: k > 10 can take minutes to run")
	flag.IntVar(&window, "w", 30, "Time window for vol prediction in number of bars, minimum 30")
	flag.IntVar(&samples, "n", 200, "Number of simulations for computing mean and std dev of vol estimate, minimum 100")
	flag.StringVar(&infile, "i", "", "Name of csv file containing return data")
	flag.StringVar(&outfile, "o", "results.csv", "Name of output file")
}

func main() {

	flag.Parse()
	if (infile == "") || (msmdim < 1) {
		flag.PrintDefaults()
		os.Exit(-1)
	}

	// read log returns in
	x := getData(infile)
	// calculate mean and std dev of returns
	// deamean returns if required
	mu, _ := stat.MeanStdDev(x, nil)
	if mu != 0.0 {
		for i := range x {
			x[i] -= mu
		}
	}

	// Fit MSM-BT model to data
	res := fit(x, msmdim)

	// Predict vol
	vol := predict(res[:len(res)-1], samples, window)

	// Write results to file
	writeResults(res, msmdim, vol, outfile)
	fmt.Printf("Wrote results to %s\n", outfile)
}

// Read returns data stored in CSV format
func getData(filename string) []float64 {
	f, err := os.Open(filename)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	rawData, err := reader.ReadAll()
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
	d := 0.0
	var x []float64
	for i := 0; i < len(rawData); i++ {
		d, err = strconv.ParseFloat(rawData[i][0], 64)
		x = append(x, d)
	}
	return x
}

// Write results to a csv file
func writeResults(res []float64, k int, vol []float64, filename string) {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
	defer f.Close()

	w := csv.NewWriter(f)

	// Prepare header
	m := len(res) + 1 + len(vol)
	hdrs := make([]string, m)
	hdrs[0], hdrs[1] = "m0", "s0"
	for i := 2; i < m-4; i++ {
		hdrs[i] = "p" + strconv.FormatInt(int64(i-1), 10)
	}
	hdrs[m-4] = "nll"
	hdrs[m-3] = "k"
	hdrs[m-2] = "volmean"
	hdrs[m-1] = "volsd"

	// Prepare data
	dat := make([]string, m)
	for i := range res {
		dat[i] = strconv.FormatFloat(res[i], 'f', -1, 64)
	}
	dat[m-3] = strconv.FormatInt(int64(k), 10)
	dat[m-2] = strconv.FormatFloat(vol[0], 'f', -1, 64)
	dat[m-1] = strconv.FormatFloat(vol[1], 'f', -1, 64)
	// Record to be written
	records := make([][]string, 2)
	records[0] = hdrs
	records[1] = dat
	w.WriteAll(records)
	if err := w.Error(); err != nil {
		fmt.Println(err)
	}
}
