package next

import (
	"fmt"
	"os"

	"github.com/hiro4bbh/sticker/sticker-util/common"
)

type Command interface {
	Parse(args []string) ([]string, error)
	Run() error
}

type TestCommand interface {
	Command
}

var newTestCommand = func(opts common.Options) TestCommand {
	panic("unsupported @testNext")
}

func NewTestCommand(opts common.Options) TestCommand {
	return newTestCommand(opts)
}

type TrainCommand interface {
	Command
}

var newTrainCommand = func(opts common.Options) TrainCommand {
	panic("unsupported @trainNext")
}

func NewTrainCommand(opts common.Options) TrainCommand {
	return newTrainCommand(opts)
}

func ReadLabelNext(filename string) (*LabelNext, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("ReadLabelNext: %s: %s", filename, err)
	}
	defer file.Close()
	var model LabelNext
	if err := DecodeLabelNext(&model, file); err != nil {
		return nil, fmt.Errorf("ReadLabelNext: %s: %s", filename, err)
	}
	return &model, nil
}
