package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/hiro4bbh/sticker"
	"github.com/hiro4bbh/sticker/sticker-util/common"
)

// TrainNearestCommand have flags for trainBoost sub-command.
type TrainNearestCommand struct {
	Help       bool
	TableNames common.OptionStrings

	opts    *Options
	flagSet *flag.FlagSet
}

// NewTrainNearestCommand returns a new TrainNearestCommand.
func NewTrainNearestCommand(opts *Options) *TrainNearestCommand {
	return &TrainNearestCommand{
		Help:       false,
		TableNames: common.OptionStrings{true, []string{"train.txt"}},
		opts:       opts,
	}
}

func (cmd *TrainNearestCommand) initializeFlagSet() {
	cmd.flagSet = flag.NewFlagSet("@trainNearest", flag.ContinueOnError)
	cmd.flagSet.Usage = func() {}
	cmd.flagSet.SetOutput(ioutil.Discard)
	cmd.flagSet.BoolVar(&cmd.Help, "help", cmd.Help, "Show the help and exit")
	cmd.flagSet.Var(&cmd.TableNames, "table", "Specify the table names")
}

// Parse parses the flags in args, and returns the remain parts of args.
//
// This function returns an error in parsing.
func (cmd *TrainNearestCommand) Parse(args []string) ([]string, error) {
	cmd.initializeFlagSet()
	if err := cmd.flagSet.Parse(args); err != nil {
		return nil, err
	}
	return cmd.flagSet.Args(), nil
}

// Run trains on the specified table of dataset.
func (cmd *TrainNearestCommand) Run() error {
	if cmd.Help {
		cmd.ShowHelp()
		return nil
	}
	opts := cmd.opts
	opts.Logger.Printf("TrainNearestCommands: %#v", cmd)
	dsname := opts.GetDatasetName()
	ds := &sticker.Dataset{
		X: sticker.FeatureVectors{},
		Y: sticker.LabelVectors{},
	}
	if len(cmd.TableNames.Values) == 0 {
		return fmt.Errorf("specify table names")
	}
	for _, tblname := range cmd.TableNames.Values {
		opts.Logger.Printf("loading table %q of dataset %q ...", tblname, dsname)
		subds, err := opts.ReadDataset(tblname)
		if err != nil {
			return err
		}
		ds.X, ds.Y = append(ds.X, subds.X...), append(ds.Y, subds.Y...)
	}
	model, err := sticker.TrainLabelNearest(ds, opts.DebugLogger)
	if err != nil {
		return err
	}
	joinedTblname := common.JoinTableNames(cmd.TableNames.Values)
	filename := opts.LabelNearest
	if filename == "" {
		filename = fmt.Sprintf("./labelnearest/%s.%s.labelnearest", dsname, joinedTblname)
		opts.LabelNearest = filename
	}
	dirpath := filepath.Dir(filename)
	if err := os.MkdirAll(dirpath, os.ModePerm); err != nil {
		return fmt.Errorf("%s: %s", dirpath, err)
	}
	opts.Logger.Printf("writing the model to %s ...", filename)
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("%s: %s", filename, err)
	}
	defer file.Close()
	if err := sticker.EncodeLabelNearest(model, file); err != nil {
		return fmt.Errorf("%s: %s", filename, err)
	}
	return nil
}

// ShowHelp shows the help.
func (cmd *TrainNearestCommand) ShowHelp() {
	fmt.Fprintf(cmd.opts.ErrorWriter, "sticker-util\nCopyright 2017- Tatsuhiro Aoshima (hiro4bbh@gmail.com).\n\nUsage: @trainNearest [subCommandOptions]\n")
	if cmd.flagSet == nil {
		cmd.initializeFlagSet()
	}
	cmd.flagSet.SetOutput(cmd.opts.ErrorWriter)
	cmd.flagSet.PrintDefaults()
	cmd.flagSet.SetOutput(ioutil.Discard)
}
