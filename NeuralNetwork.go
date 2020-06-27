package main

import (
	"log"

	"github.com/NOX73/go-neural"
	"github.com/NOX73/go-neural/learn"
)

func NeuralNetwork() {

	n := neural.NewNetwork(3, []int{9, 9, 2})
	// Randomize sypaseses weights
	n.RandomizeSynapses()

	var trainings []Training
	trainings = []Training{
		Training{
			Input:  []float64{1, 2, 3},
			Output: []float64{0, 1},
		},
		Training{
			Input:  []float64{4, 2, 3},
			Output: []float64{1, 0},
		},
		Training{
			Input:  []float64{4, 5, 3},
			Output: []float64{1, 0},
		},
		Training{
			Input:  []float64{4, 5, 6},
			Output: []float64{1, 0},
		},
	}
	// Learning speed [0..1]
	var speed float64
	speed = 1

	// engine := engine.New(n)
	// engine.Start()
	for i := 0; i <= 100000; i++ {
		for _, train := range trainings {
			learn.Learn(n, train.Input, train.Output, speed)
		}
	}

	e := learn.Evaluation(n, trainings[0].Input, trainings[0].Output)

	result := n.Calculate([]float64{4, 5, 3})
	log.Printf("%+v | %+v", result, e)
}
