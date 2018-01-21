package common

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/hiro4bbh/sticker"
	"github.com/hiro4bbh/sticker/plugin"
)

// Options is an interface for sticker-util.Options.
type Options interface {
	FeatureMap(feature uint32, quote bool) string
	GetDatasetName() string
	GetDebugLogger() *log.Logger
	GetErrorWriter() io.Writer
	GetLabelNext() string
	GetLogger() *log.Logger
	GetOutputWriter() io.Writer
	LabelMap(label uint32, quote bool) string
	ReadDataset(tblname string) (*sticker.Dataset, error)
	SetLabelNext(labelNext string)
}

// JoinTableNames returns the joined table name from the given table names cut its file extension.
func JoinTableNames(tblnames []string) string {
	name := ""
	for i, tblname := range tblnames {
		if i > 0 {
			name += "_"
		}
		name += strings.TrimSuffix(tblname, filepath.Ext(tblname))
	}
	return name
}

// ReadLabelBoost reads the .labelboost model file.
func ReadLabelBoost(filename string) (*plugin.LabelBoost, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("ReadLabelBoost: %s: %s", filename, err)
	}
	defer file.Close()
	var model plugin.LabelBoost
	if err := plugin.DecodeLabelBoost(&model, file); err != nil {
		return nil, fmt.Errorf("ReadLabelBoost: %s: %s", filename, err)
	}
	return &model, nil
}

// ReadLabelConst reads the .labelconst model file.
func ReadLabelConst(filename string) (*sticker.LabelConst, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("ReadLabelConst: %s: %s", filename, err)
	}
	defer file.Close()
	var model sticker.LabelConst
	if err := sticker.DecodeLabelConst(&model, file); err != nil {
		return nil, fmt.Errorf("ReadLabelConst: %s: %s", filename, err)
	}
	return &model, nil
}

// ReadLabelForest reads the .labelforest model file.
func ReadLabelForest(filename string) (*plugin.LabelForest, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("ReadLabelForest: %s: %s", filename, err)
	}
	defer file.Close()
	var forest plugin.LabelForest
	if err := plugin.DecodeLabelForest(&forest, file); err != nil {
		return nil, fmt.Errorf("ReadLabelForest: %s: %s", filename, err)
	}
	return &forest, nil
}

// ReadLabelNearest reads the .labelnearest model file.
func ReadLabelNearest(filename string) (*sticker.LabelNearest, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("ReadLabelNearest: %s: %s", filename, err)
	}
	defer file.Close()
	var model sticker.LabelNearest
	if err := sticker.DecodeLabelNearest(&model, file); err != nil {
		return nil, fmt.Errorf("ReadLabelNearest: %s: %s", filename, err)
	}
	return &model, nil
}

// ReadLabelOne reads the .labelone model file.
func ReadLabelOne(filename string) (*sticker.LabelOne, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("ReadLabelOne: %s: %s", filename, err)
	}
	defer file.Close()
	var model sticker.LabelOne
	if err := sticker.DecodeLabelOne(&model, file); err != nil {
		return nil, fmt.Errorf("ReadLabelOne: %s: %s", filename, err)
	}
	return &model, nil
}

// OptionFloat32 is the data structure for using float32 in flag.
// This implements interface flag.Value.
type OptionFloat32 float32

// String is for interface flag.Value.
// The default value is "0.0".
func (opt *OptionFloat32) String() string {
	return "0.0"
}

// Set is for interface flag.Value.
func (opt *OptionFloat32) Set(value string) error {
	valueF64, err := strconv.ParseFloat(value, 32)
	if err != nil {
		return err
	}
	*opt = OptionFloat32(float32(valueF64))
	return nil
}

// OptionStrings is the data structure for using string slice in flag.
// This implements interface flag.Value.
type OptionStrings []string

// String is for interface flag.Value.
// The default value is "".
func (opt *OptionStrings) String() string {
	return ""
}

// Set is for interface flag.Value.
func (opt *OptionStrings) Set(value string) error {
	*opt = append(*opt, value)
	return nil
}

// OptionUints is the data structure for using uint slice in flag.
// This implements interface flag.Value.
type OptionUints []uint

// String is for interface flag.Value.
// The default value is "".
func (opt *OptionUints) String() string {
	return ""
}

// Set is for interface flag.Value.
func (opt *OptionUints) Set(value string) error {
	valueUint, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return err
	}
	*opt = append(*opt, uint(valueUint))
	return nil
}
