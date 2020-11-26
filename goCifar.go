package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"runtime/pprof"
	"syscall"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"time"

	"./cifar"

	"gopkg.in/cheggaaa/pb.v1"
)

var (
	epochs     = flag.Int("epochs", 25, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Cifar \"train\" or \"test\"")
	dtype      = flag.String("dtypeype", "float64", "Which dtypeype to use")
	batch  = flag.Int("batch", 100, "mini Batch")
	
)

const directory = "./cifar-10/"

var dtyp tensor.Dtype

func getdtype() {
	switch *dtype {
	case "float64":
		dtyp = tensor.Float64
	case "float32":
		dtyp = tensor.Float32
	default:
		log.Fatalf("Unknown dtyp")
	}
}

type rangeStep struct {
	start, end int
}

func (r rangeStep) Start() int { return r.start }
func (r rangeStep) End() int   { return r.end }
func (r rangeStep) Step() int  { return 1 }

type conv2d struct {
	graph   *gorgonia.ExprGraph
	weight0, weight1, weight2, weight3, weight4 *gorgonia.Node // weights
	dropout   float64        // dropouts

	out     *gorgonia.Node
	prediction gorgonia.Value
}
//cirfar data 3 channels 5x5 image size out channels 32,64,128,512  num classes 10 init weights with Glorot

func newconv2d(graph *gorgonia.ExprGraph) *conv2d {
	weight0 := gorgonia.NewTensor(graph, dtyp, 4, gorgonia.WithShape(32, 3, 5, 5), gorgonia.WithName("weight0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	weight1 := gorgonia.NewTensor(graph, dtyp, 4, gorgonia.WithShape(64, 32, 5, 5), gorgonia.WithName("weight1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	weight2 := gorgonia.NewTensor(graph, dtyp, 4, gorgonia.WithShape(128, 64, 5, 5), gorgonia.WithName("weight2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	weight3 := gorgonia.NewMatrix(graph, dtyp, gorgonia.WithShape(512, 256), gorgonia.WithName("weight3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	weight4 := gorgonia.NewMatrix(graph, dtyp, gorgonia.WithShape(256, 10), gorgonia.WithName("weight4"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &conv2d{
		graph:  graph,
		weight0: weight0,
		weight1: weight1,
		weight2: weight2,
		weight3: weight3,
		weight4: weight4,

		dropout: 0.2,
	}
}

func (cnn *conv2d) parameters() gorgonia.Nodes {
	return gorgonia.Nodes{cnn.weight0, cnn.weight1, cnn.weight2, cnn.weight3, cnn.weight4}
}

func (cnn *conv2d) foward(inp *gorgonia.Node) (err error) {
	var cnn0, cnn1, cnn2, fc *gorgonia.Node
	var activation0, activation1, activation2, activation3 *gorgonia.Node
	var pool0, pool1, pool2 *gorgonia.Node
	var layer0, layer1, layer2, layer3 *gorgonia.Node

	if cnn0, err = gorgonia.Conv2d(inp, cnn.weight0, tensor.Shape{5, 5}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 0 Convolution failed")
	}
	if activation0, err = gorgonia.Rectify(cnn0); err != nil {
		return errors.Wrap(err, "Layer 0 activation failed")
	}
	if pool0, err = gorgonia.MaxPool2D(activation0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 0 Maxpooling failed")
	}
	log.Printf("pool0 shape %v", pool0.Shape())
	if layer0, err = gorgonia.Dropout(pool0, cnn.dropout); err != nil {  //dropout
		return errors.Wrap(err, "Dropout error")
	}

	// Layer 1
	if cnn1, err = gorgonia.Conv2d(layer0, cnn.weight1, tensor.Shape{5, 5}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if activation1, err = gorgonia.Rectify(cnn1); err != nil {
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	if pool1, err = gorgonia.MaxPool2D(activation1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 1 Maxpooling failed")
	}
	if layer1, err = gorgonia.Dropout(pool1, cnn.dropout); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout to layer 1")
	}

	// Layer 2
	if cnn2, err = gorgonia.Conv2d(layer1, cnn.weight2, tensor.Shape{5, 5}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 2 Convolution failed")
	}
	if activation2, err = gorgonia.Rectify(cnn2); err != nil {  //activation
		return errors.Wrap(err, "Layer 2 activation failed")
	}
	//max pool layer
	if pool2, err = gorgonia.MaxPool2D(activation2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 2 Maxpooling failed")
	}
	log.Printf("pool2 shape %v", pool2.Shape())

	var node *gorgonia.Node
	temp, temp1, temp2, temp3 := pool2.Shape()[0], pool2.Shape()[1], pool2.Shape()[2], pool2.Shape()[3]
	if node, err = gorgonia.Reshape(pool2, tensor.Shape{temp, temp1 * temp2 * temp2}); err != nil {
		return errors.Wrap(err, "Unable to reshape layer 2")
	}
	log.Printf("node shape %v", node.Shape())
	if layer2, err = gorgonia.Dropout(node, cnn.dropout); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout on layer 2")
	}

	ioutil.WriteFile("temp.dot", []byte(cnn.graph.ToDot()), 0644)

	// Layer 3
	if fc, err = gorgonia.Mul(layer2, cnn.weight3); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if activation3, err = gorgonia.Rectify(fc); err != nil {
		return errors.Wrapf(err, "Unable to activate fc")
	}
	if layer3, err = gorgonia.Dropout(activation3, cnn.droptout); err != nil {
		return errors.Wrapf(err, "Unable to apply a dropout on layer 3")
	}


	var out *gorgonia.Node
	if out, err = gorgonia.Mul(layer3, cnn.weight4); err != nil {
		return errors.Wrapf(err, "matrix multiply error")
	}
	cnn.out, err = gorgonia.SoftMax(out)  //apply softmax
	gorgonia.Read(cnn.out, &cnn.prediction)
	return
}

func main() {
	flag.Parse()
	getdtype()
	rand.Seed(2020)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	doneChan := make(chan bool, 1)

	var inputs, targets tensor.Tensor
	var err error


	train := *dataset
	if inputs, targets, err = cifar.Load(train, directory); err != nil {
		log.Fatal(err)
	}

	numExamples := inputs.Shape()[0]
	miniBatch := *batch

	if err := inputs.Reshape(numExamples, 3, 32, 32); err != nil {
		log.Fatal(err)
	}
	grp := gorgonia.NewGraph()
	x := gorgonia.NewTensor(grp, dtype, 4, gorgonia.WithShape(miniBatch, 3, 32, 32), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(grp, dtype, gorgonia.WithShape(miniBatch, 10), gorgonia.WithName("y"))
	conv := newconv2d(grp)
	if err = conv.foward(inp); err != nil {
		log.Fatalf("%+v", err)
	}

	//Hadamard Product - element wise
	losses := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Log(conv.out)), y))
	cost := gorgonia.Must(gorgonia.Sum(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	var accuracy float64

	if _, err = gorgonia.Grad(cost, conv.parameters()...); err != nil {
		log.Fatal(err)
	}

	ioutil.WriteFile("GraphOps.dot", []byte(grp.ToDot()), 0644)
	

	prog, directoryMap, _ := gorgonia.Compile(grp)
	log.Printf("%v", prog)

	vmach := gorgonia.NewTapeMachine(grp, gorgonia.WithPrecompiled(prog, directoryMap), gorgonia.BindDualValues(m.parameters()...))
	optimizer := gorgonia.NewRMSPropoptimizer(gorgonia.WithBatchSize(float64(miniBatch)))
	defer vmach.Close()

	
	go purge(sig, doneChan)

	steps := numExamples / miniBatch
	log.Printf("Batches %d", batches)
	progress := pb.New(steps)
	progress.SetRefreshRate(time.Second)
	progress.SetMaxWidth(80)

	//training step

	for i := 0; i < *epochs; i++ {
		progress.Prefix(fmt.Sprintf("Epoch %d", i))
		progress.Set(0)
		progress.Start()

		accuracy = 0

		for b := 0; b < steps; b++ {
			start := b * miniBatch
			end := start + miniBatch
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.rangeStepce(rangeStep{start, end}); err != nil {
				log.Fatal("Inputs error")
			}

			if yVal, err = targets.rangeStepce(rangeStep{start, end}); err != nil {
				log.Fatal("yVal error")
			}
			if err = xVal.(*tensor.Dense).Reshape(miniBatch, 3, 32, 32); err != nil {
				log.Fatalf("Dense shape mismatch %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vmach.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}

			
			arrayOut := cnn.prediction.Data().([]float64)
			yHat := tensor.New(tensor.WithShape(miniBatch, 10), tensor.WithBacking(arrayOut))

			for j := 0; j < yVal.Shape()[0]; j++ {

				// get label
				yRowT, _ := yVal.Slice(rangeStep{j, j + 1})
				yRow := yRowT.Data().([]float64)
				var rowLabel int
				var yRowHigh float64

				for k := 0; k < 10; k++ {
					if k == 0 {
						rowLabel = 0
						yRowHigh = yRow[k]
					} else if yRow[k] > yRowHigh {
						rowLabel = k
						yRowHigh = yRow[k]
					}
				}

				// get prediction
				predRowT, _ := yHat.Slice(rangeStep{j, j + 1})
				predRow := predRowT.Data().([]float64)
				var row int
				var pred float64

				// guess result
				for k := 0; k < 10; k++ {
					if k == 0 {
						rowGuess = 0
						predRowHigh = predRow[k]
					} else if predRow[k] > predRowHigh {
						row = k
						pred = predRow[k]
					}
				}

				if rowLabel == row {
					accuracy += 1.0 / float64(numExamples)
				}
			}

			// end temp

			optimizer.Step(gorgonia.NodesToValueGrads(cnn.parameters()))
			vmach.Reset()
			progress.Increment()
		}
		log.Printf("Epoch %d | cost %v | acc %v", i, costVal, accuracy)

	}

	// import test data and run more loops
	if inputs, targets, err = cifar.Load("test", directory); err != nil {
		log.Fatal(err)
	}

	batches = inputs.Shape()[0] / miniBatch
	progress = pb.New(batches)
	progress.SetRefreshRate(time.Second)
	progress.SetMaxWidtypeh(70)

	var testActual, testPred []int

	for i := 0; i < 1; i++ {
		progress.Prefix(fmt.Sprintf("Epoch Test"))
		progress.Set(0)
		progress.Start()
		for b := 0; b < batches; b++ {
			start := b * miniBatch
			end := start + miniBatch
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(rangeStep{start, end}); err != nil {
				log.Fatal("Unable to rangeStepce x")
			}

			if yVal, err = targets.Slice(rangeStep{start, end}); err != nil {
				log.Fatal("Unable to rangeStepce y")
			}
			if err = xVal.(*tensor.Dense).Reshape(miniBatch, 3, 32, 32); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vmach.RunAll(); err != nil {
				log.Fatalf("Failed at epoch test: %v", err)
			}

			arrayOut := cnn.prediction.Data().([]float64)
			yHat := tensor.New(tensor.WithShape(miniBatch, 10), tensor.WithBacking(arrayOut))

			for j := 0; j < yVal.Shape()[0]; j++ {

				// get label
				yRowT, _ := yVal.Slice(rangeStep{j, j + 1})
				yRow := yRowT.Data().([]float64)
				var rowLabel int
				var yRowHigh float64

				for k := 0; k < 10; k++ {
					if k == 0 {
						rowLabel = 0
						yRowHigh = yRow[k]
					} else if yRow[k] > yRowHigh {
						rowLabel = k
						yRowHigh = yRow[k]
					}
				}

				// get prediction
				predRowT, _ := yHat.Slice(rangeStep{j, j + 1})
				predRow := predRowT.Data().([]float64)
				var row int
				var pred float64

				// guess result
				for k := 0; k < 10; k++ {
					if k == 0 {
						row = 0
						pred = predRow[k]
					} else if predRow[k] > pred {
						row = k
						pred = predRow[k]
					}
				}

				testActual = append(testActual, rowLabel)
				testPred = append(testPred, row)
			}

			vmach.Reset()
			progress.Increment()
		}
		log.Printf("Epoch Test | cost %v", costVal)

		printSlice("testActual.csv", testActual)
		printSlice("testPred.csv", testPred)

	}
}

func purge(sig chan os.Signal, doneChan chan bool) {
	select {
	case <-sig:
		log.Println("EMERGENCY EXIT!")
		
		os.Exit(1)

	case <-doneChan:
		return
	}
}



func printSlice(filePath string, values []int) error {
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, value := range values {
		fmt.Fprintln(f, value)
	}
	return nil
}