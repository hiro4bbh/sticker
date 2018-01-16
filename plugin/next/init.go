package next

import (
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
