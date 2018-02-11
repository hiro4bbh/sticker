package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"sort"
	"time"

	"github.com/hiro4bbh/sticker"
	"github.com/hiro4bbh/sticker/sticker-util/common"
)

// TestNearestCommand have flags for testNearest sub-command.
type TestNearestCommand struct {
	Alpha      common.OptionFloat32
	Beta       common.OptionFloat32
	Help       bool
	Ks         common.OptionUints
	N          uint
	Per        uint
	S          uint
	TableNames common.OptionStrings

	opts    *Options
	flagSet *flag.FlagSet
}

// NewTestNearestCommand returns a new TestNearestCommand.
func NewTestNearestCommand(opts *Options) *TestNearestCommand {
	return &TestNearestCommand{
		Alpha:      common.OptionFloat32(1.0),
		Beta:       common.OptionFloat32(1.0),
		Help:       false,
		Ks:         common.OptionUints{true, []uint{1, 3, 5}},
		N:          ^uint(0),
		Per:        uint(0),
		S:          uint(1),
		TableNames: common.OptionStrings{true, []string{"test.txt"}},
		opts:       opts,
	}
}

func (cmd *TestNearestCommand) initializeFlagSet() {
	cmd.flagSet = flag.NewFlagSet("@testNearest", flag.ContinueOnError)
	cmd.flagSet.Usage = func() {}
	cmd.flagSet.SetOutput(ioutil.Discard)
	cmd.flagSet.Var(&cmd.Alpha, "alpha", "Specify the smoothing parameter for weighting the voted by each neighbour")
	cmd.flagSet.Var(&cmd.Beta, "beta", "Specify the smoothing parameter for balancing the Jaccard similarity and the cosine similarity")
	cmd.flagSet.BoolVar(&cmd.Help, "help", cmd.Help, "Show the help and exit")
	cmd.flagSet.Var(&cmd.Ks, "K", "Specify the top-K values")
	cmd.flagSet.UintVar(&cmd.N, "N", cmd.N, "Specify the maximum number of the tested entries")
	cmd.flagSet.UintVar(&cmd.Per, "per", cmd.Per, "Specify the deep-inspection timing counts (not do deep-inspection if 0)")
	cmd.flagSet.UintVar(&cmd.S, "S", cmd.S, "Specify the number of nearest neighbours")
	cmd.flagSet.Var(&cmd.TableNames, "table", "Specify the table names")
}

// Parse parses the flags in args, and returns the remain parts of args.
//
// This function returns an error in parsing.
func (cmd *TestNearestCommand) Parse(args []string) ([]string, error) {
	cmd.initializeFlagSet()
	if err := cmd.flagSet.Parse(args); err != nil {
		return nil, err
	}
	return cmd.flagSet.Args(), nil
}

// Run tests the .labelforest model on the specified table of dataset.
func (cmd *TestNearestCommand) Run() error {
	if cmd.Help {
		cmd.ShowHelp()
		return nil
	}
	rng := rand.New(rand.NewSource(0))
	opts := cmd.opts
	opts.Logger.Printf("TestNearestCommands: %#v", cmd)
	dsname := opts.GetDatasetName()
	ds := &sticker.Dataset{
		X: sticker.FeatureVectors{},
		Y: sticker.LabelVectors{},
	}
	if len(cmd.TableNames.Values) == 0 {
		return fmt.Errorf("specify the table names")
	}
	for _, tblname := range cmd.TableNames.Values {
		opts.Logger.Printf("loading table %q of dataset %q ...", tblname, dsname)
		subds, err := opts.ReadDataset(tblname)
		if err != nil {
			return err
		}
		ds.X, ds.Y = append(ds.X, subds.X...), append(ds.Y, subds.Y...)
	}
	if cmd.N < uint(ds.Size()) {
		for i := 0; i < int(cmd.N); i++ {
			j := i + rng.Intn(ds.Size()-i)
			ds.X[i], ds.X[j], ds.Y[i], ds.Y[j] = ds.X[j], ds.X[i], ds.Y[j], ds.Y[i]
		}
		ds.X, ds.Y = ds.X[:cmd.N], ds.Y[:cmd.N]
	}
	opts.Logger.Printf("loading .labelnearest model from %q ...", opts.LabelNearest)
	model, err := common.ReadLabelNearest(opts.LabelNearest)
	if err != nil {
		return err
	}
	maxK := uint(0)
	for _, K := range cmd.Ks.Values {
		if maxK < K {
			maxK = K
		}
	}
	maxAvgPrecisions := make([]float32, 0, len(cmd.Ks.Values))
	for _, K := range cmd.Ks.Values {
		maxPrecisionKs := sticker.ReportMaxPrecision(ds.Y, K)
		maxSumPrecisionK := float32(0.0)
		for _, maxPrecisionKi := range maxPrecisionKs {
			maxSumPrecisionK += maxPrecisionKi
		}
		maxAvgPrecisions = append(maxAvgPrecisions, maxSumPrecisionK/float32(len(ds.Y)))
	}
	opts.Logger.Printf("predicting top-%d labels ...", maxK)
	inferenceStartTime := time.Now()
	precisionKsMap, nDCGKsMap := make(map[uint][]float32), make(map[uint][]float32)
	for _, K := range cmd.Ks.Values {
		precisionKsMap[K], nDCGKsMap[K] = make([]float32, 0, ds.Size()), make([]float32, 0, ds.Size())
	}
	report := func(Y_ sticker.LabelVectors, flush bool) {
		n := len(Y_)
		inferenceEndTime := time.Now()
		inferenceTime := inferenceEndTime.Sub(inferenceStartTime)
		inferenceTimePerEntry := time.Duration(inferenceTime.Nanoseconds() / int64(n)).Round(time.Microsecond)
		fmt.Fprintf(opts.OutputWriter, "finished inference on %d/%d entries (%-5.4g%%) in %s (about %s/entry)\n", n, len(ds.Y), float32(n)/float32(len(ds.Y))*100.0, inferenceTime, inferenceTimePerEntry)
		precisions, nDCGs := make([]float32, 0, len(cmd.Ks.Values)), make([]float32, 0, len(cmd.Ks.Values))
		for iK, K := range cmd.Ks.Values {
			startidx := len(precisionKsMap[K])
			precisionKsMap[K] = append(precisionKsMap[K], sticker.ReportPrecision(ds.Y[startidx:n], K, Y_[startidx:n])...)
			sumPrecisionK := float32(0.0)
			for _, precisionKi := range precisionKsMap[K] {
				sumPrecisionK += precisionKi
			}
			avgPrecisionK := sumPrecisionK / float32(n)
			nDCGKsMap[K] = append(nDCGKsMap[K], sticker.ReportNDCG(ds.Y[startidx:n], K, Y_[startidx:n])...)
			sumNDCGK := float32(0.0)
			for _, nDCGKi := range nDCGKsMap[K] {
				sumNDCGK += nDCGKi
			}
			avgNDCGK := sumNDCGK / float32(n)
			fmt.Fprintf(opts.OutputWriter, "Precision@%d=%-5.4g%%/%-5.4g%%, nDCG@%d=%-5.4g%%\n", K, avgPrecisionK*100, maxAvgPrecisions[iK]*100, K, avgNDCGK*100)
			if flush {
				precisions = append(precisions, avgPrecisionK)
				nDCGs = append(nDCGs, avgNDCGK)
			}
		}
	}
	if cmd.Per == 0 {
		report(model.PredictAll(ds.X, maxK, cmd.S, float32(cmd.Alpha), float32(cmd.Beta)), true)
	} else {
		Y_ := make(sticker.LabelVectors, 0, ds.Size())
		for i, xi := range ds.X {
			yi_, labelHist, indexSimsTopS := model.Predict(xi, maxK, cmd.S, float32(cmd.Alpha), float32(cmd.Beta))
			Y_ = append(Y_, yi_)
			if uint(i)%cmd.Per == 0 {
				if opts.DebugLogger != nil {
					yimap := make(map[uint32]int)
					for _, label := range ds.Y[i] {
						yimap[label] = 30
					}
					lenxi := float32(0.0)
					for _, xipair := range xi {
						lenxi += xipair.Value * xipair.Value
					}
					lenxi = sticker.Sqrt32(lenxi)
					opts.DebugLogger.Printf("i=%d, ||xi||_2=%.3g", i, lenxi)
					line := "  indexSimsTopS=map["
					for k, indexSim := range indexSimsTopS {
						if k > 0 {
							line += " "
						}
						line += fmt.Sprintf("%s:%.3g", opts.LabelMap(indexSim.Key, true), indexSim.Value)
					}
					line += "]"
					opts.DebugLogger.Print(line)
					line = "  labelHist=map["
					labelHistKV := make(sticker.KeyValues32OrderedByValue, 0, len(labelHist))
					for label, freq := range labelHist {
						labelHistKV = append(labelHistKV, sticker.KeyValue32{label, freq})
					}
					sort.Sort(sort.Reverse(labelHistKV))
					for rank, labelFreq := range labelHistKV {
						if rank > 0 {
							line += " "
						}
						colorStart, colorEnd := "", ""
						if _, ok := yimap[labelFreq.Key]; ok {
							if uint(rank) < maxK {
								colorStart, colorEnd = "\033[32m", "\033[0m"
								yimap[labelFreq.Key] = 32
							} else {
								colorStart, colorEnd = "\033[36m", "\033[0m"
								yimap[labelFreq.Key] = 36
							}
						}
						line += fmt.Sprintf("%s%s%s:%.3g", colorStart, opts.LabelMap(labelFreq.Key, true), colorEnd, labelFreq.Value)
					}
					line += "]"
					opts.DebugLogger.Print(line)
					line = "  <> yi=["
					for k, label := range ds.Y[i] {
						if k > 0 {
							line += " "
						}
						line += fmt.Sprintf("\033[%dm%s\033[0m", yimap[label], opts.LabelMap(label, true))
					}
					line += "]: "
					for iK, K := range cmd.Ks.Values {
						if iK > 0 {
							line += ","
						}
						pK := sticker.ReportPrecision(sticker.LabelVectors{ds.Y[i]}, K, sticker.LabelVectors{yi_})
						line += fmt.Sprintf("P@%d=%.2f%%", K, pK[0]*100)
					}
					opts.DebugLogger.Print(line)
				}
				report(Y_, false)
			}
		}
		report(Y_, true)
	}
	return nil
}

// ShowHelp shows the help.
func (cmd *TestNearestCommand) ShowHelp() {
	fmt.Fprintf(cmd.opts.ErrorWriter, "sticker-util\nCopyright 2017- Tatsuhiro Aoshima (hiro4bbh@gmail.com).\n\nUsage: @testNearest [subCommandOptions]\n")
	if cmd.flagSet == nil {
		cmd.initializeFlagSet()
	}
	cmd.flagSet.SetOutput(cmd.opts.ErrorWriter)
	cmd.flagSet.PrintDefaults()
	cmd.flagSet.SetOutput(ioutil.Discard)
}
