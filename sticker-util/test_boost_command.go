package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"github.com/hiro4bbh/sticker"
	"github.com/hiro4bbh/sticker/sticker-util/common"
)

// TestBoostCommand have flags for testBoost sub-command.
type TestBoostCommand struct {
	Help       bool
	Ks         common.OptionUints
	Restore    bool
	Ts         common.OptionUints
	TableNames common.OptionStrings

	Result map[string]interface{}

	opts    *Options
	flagSet *flag.FlagSet
}

// NewTestBoostCommand returns a new TestBoostCommand.
func NewTestBoostCommand(opts *Options) *TestBoostCommand {
	return &TestBoostCommand{
		Help:       false,
		Ks:         common.OptionUints{},
		Restore:    false,
		Ts:         common.OptionUints{},
		TableNames: common.OptionStrings{},
		opts:       opts,
	}
}

func (cmd *TestBoostCommand) initializeFlagSet() {
	cmd.flagSet = flag.NewFlagSet("@testBoost", flag.ContinueOnError)
	cmd.flagSet.Usage = func() {}
	cmd.flagSet.SetOutput(ioutil.Discard)
	cmd.flagSet.BoolVar(&cmd.Help, "help", cmd.Help, "Show the help and exit")
	cmd.flagSet.Var(&cmd.Ks, "K", "Specify the top-K values")
	cmd.flagSet.BoolVar(&cmd.Restore, "restore", cmd.Restore, "Restore the test result if true")
	cmd.flagSet.Var(&cmd.Ts, "T", "Specify the used numbers of rounds (use all rounds if zero)")
	cmd.flagSet.Var(&cmd.TableNames, "table", "Specify the table names")
}

// Parse parses the flags in args, and returns the remain parts of args.
//
// This function returns an error in parsing.
func (cmd *TestBoostCommand) Parse(args []string) ([]string, error) {
	cmd.initializeFlagSet()
	if err := cmd.flagSet.Parse(args); err != nil {
		return nil, err
	}
	if len(cmd.Ts) == 0 {
		cmd.Ts = common.OptionUints{0}
	}
	return cmd.flagSet.Args(), nil
}

// Run tests the .labelforest model on the specified table of dataset.
func (cmd *TestBoostCommand) Run() error {
	if cmd.Help {
		cmd.ShowHelp()
		return nil
	}
	opts := cmd.opts
	opts.Logger.Printf("TestBoostCommands: %#v", cmd)
	dsname := opts.GetDatasetName()
	tblname := common.JoinTableNames(cmd.TableNames)
	restoreName := fmt.Sprintf("%s.testBoost.%s.%s.bin", opts.LabelBoost, dsname, tblname)
	if cmd.Restore {
		opts.Logger.Printf("restroing the dumped test result in %q ...", restoreName)
		f, err := os.Open(restoreName)
		if err != nil {
			return err
		}
		defer f.Close()
		return gob.NewDecoder(f).Decode(&cmd.Result)
	}
	ds := &sticker.Dataset{
		X: sticker.FeatureVectors{},
		Y: sticker.LabelVectors{},
	}
	if len(cmd.TableNames) == 0 {
		return fmt.Errorf("specify the table names")
	}
	for _, tblname := range cmd.TableNames {
		opts.Logger.Printf("loading table %q of dataset %q ...", tblname, dsname)
		subds, err := opts.ReadDataset(tblname)
		if err != nil {
			return err
		}
		ds.X, ds.Y = append(ds.X, subds.X...), append(ds.Y, subds.Y...)
	}
	n := ds.Size()
	opts.Logger.Printf("loading .labelboost model from %q ...", opts.LabelBoost)
	model, err := common.ReadLabelBoost(opts.LabelBoost)
	if err != nil {
		return err
	}
	maxK := uint(0)
	for _, K := range cmd.Ks {
		if maxK < K {
			maxK = K
		}
	}
	maxAvgPrecisions := make([]float32, 0, len(cmd.Ks))
	for _, K := range cmd.Ks {
		maxPrecisionKs := sticker.ReportMaxPrecision(ds.Y, K)
		maxSumPrecisionK := float32(0.0)
		for _, maxPrecisionKi := range maxPrecisionKs {
			maxSumPrecisionK += maxPrecisionKi
		}
		maxAvgPrecisions = append(maxAvgPrecisions, maxSumPrecisionK/float32(len(ds.Y)))
	}
	rounds := make([]interface{}, 0, len(cmd.Ts))
	for _, T := range cmd.Ts {
		inferenceStartTime := time.Now()
		opts.Logger.Printf("T=%d: predicting top-%d labels ...", T, maxK)
		Y := model.PredictAll(ds.X, maxK, T)
		inferenceEndTime := time.Now()
		inferenceTime := inferenceEndTime.Sub(inferenceStartTime)
		inferenceTimePerEntry := time.Duration(inferenceTime.Nanoseconds() / int64(n)).Round(time.Microsecond)
		fmt.Fprintf(opts.OutputWriter, "T=%d: finished inference on %d entries in %s (about %s/entry)\n", T, n, inferenceTime, inferenceTimePerEntry)
		precisions, nDCGs := make([]float32, 0, len(cmd.Ks)), make([]float32, 0, len(cmd.Ks))
		for iK, K := range cmd.Ks {
			precisionKs := sticker.ReportPrecision(ds.Y, K, Y)
			sumPrecisionK := float32(0.0)
			for _, precisionKi := range precisionKs {
				sumPrecisionK += precisionKi
			}
			avgPrecisionK := sumPrecisionK / float32(len(ds.Y))
			precisions = append(precisions, avgPrecisionK)
			nDCGKs := sticker.ReportNDCG(ds.Y, K, Y)
			sumNDCGK := float32(0.0)
			for _, nDCGKi := range nDCGKs {
				sumNDCGK += nDCGKi
			}
			avgNDCGK := sumNDCGK / float32(len(ds.Y))
			nDCGs = append(nDCGs, avgNDCGK)
			fmt.Fprintf(opts.OutputWriter, "T=%d: Precision@%d=%-5.4g%%/%-5.4g%%, nDCG@%d=%-5.4g%%\n", T, K, avgPrecisionK*100, maxAvgPrecisions[iK]*100, K, avgNDCGK*100)
		}
		rounds = append(rounds, map[string]interface{}{
			"T":                     T,
			"inferenceTime":         fmt.Sprintf("%s", inferenceTime),
			"inferenceTimePerEntry": fmt.Sprintf("%s", inferenceTimePerEntry),
			"precisions":            precisions,
			"nDCGs":                 nDCGs,
		})
	}
	cmd.Result = map[string]interface{}{
		"Ks":            []uint(cmd.Ks),
		"maxPrecisions": maxAvgPrecisions,
		"nentries":      n,
		"rounds":        rounds,
	}
	opts.Logger.Printf("dumping the test result to %q ...", restoreName)
	f, err := os.Create(restoreName)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(cmd.Result)
}

// ShowHelp shows the help.
func (cmd *TestBoostCommand) ShowHelp() {
	fmt.Fprintf(cmd.opts.ErrorWriter, "sticker-util\nCopyright 2017- Tatsuhiro Aoshima (hiro4bbh@gmail.com).\n\nUsage: @testBoost [subCommandOptions]\n")
	if cmd.flagSet == nil {
		cmd.initializeFlagSet()
	}
	cmd.flagSet.SetOutput(cmd.opts.ErrorWriter)
	cmd.flagSet.PrintDefaults()
	cmd.flagSet.SetOutput(ioutil.Discard)
}
