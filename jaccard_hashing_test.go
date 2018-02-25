package sticker

import (
	"bytes"
	"encoding/gob"
	"testing"

	"github.com/hiro4bbh/go-assert"
)

func TestJaccardHashing(t *testing.T) {
	x1 := FeatureVector{KeyValue32{1, 1.0}, KeyValue32{2, 1.0}}
	x2 := FeatureVector{KeyValue32{1, 1.0}, KeyValue32{2, 1.0}, KeyValue32{5, 1.0}, KeyValue32{6, 1.0}}
	hashing := NewJaccardHashing(16, 10, 8)
	goassert.New(t, uint(16)).Equal(hashing.K())
	goassert.New(t, uint(10)).Equal(hashing.L())
	goassert.New(t, uint(8)).Equal(hashing.R())
	hashing.Add(x1, 0)
	hashing.Add(x2, 1)
	// Some errors can be permissible.
	goassert.New(t, map[uint32]uint32{0: 16, 1: 7}).Equal(hashing.FindNears(x1).Map())
	goassert.New(t, map[uint32]uint32{0: 7, 1: 16}).Equal(hashing.FindNears(x2).Map())
	goassert.New(t, []int{1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1}, []int{18, 7, 0, 0, 0, 0, 0, 0}).Equal(hashing.Summary())
	// Check the reservoir boundedness.
	for t := uint32(2); t < 128; t++ {
		hashing.Add(x1, t)
	}
	// FIXME: Heavy conflicts would decrease the estimated Jaccard similarity due to reservoir sampling.
	//goassert.New(t, map[uint32]uint32{0: 16, 1: 7}).Equal(hashing.FindNears(x1))
	goassert.New(t, []int{1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1}, []int{9, 0, 0, 0, 0, 0, 0, 16}).Equal(hashing.Summary())
	// Check gob encoding/decoding.
	var buf bytes.Buffer
	goassert.New(t).SucceedWithoutError(EncodeJaccardHashing(hashing, &buf))
	var decodedHashing JaccardHashing
	goassert.New(t).SucceedWithoutError(DecodeJaccardHashing(&decodedHashing, &buf))
	goassert.New(t, hashing).Equal(&decodedHashing)
	// gob.Decoder.Decode won't call JaccardHashing.GobDecode, because the encoder did not encode JaccardHashing.
	goassert.New(t, "JaccardHashing should be encoded with EncodeJaccardHashing").ExpectError(gob.NewEncoder(&buf).Encode(&decodedHashing))
	// Check that an empty FeatureVector can be hashed safely.
	hashing0 := NewJaccardHashing(16, 10, 8)
	hashing0.Add(FeatureVector{}, 0)
	goassert.New(t, []uint32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}).Equal(hashing0.Hash(FeatureVector{}))
	goassert.New(t, []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, []int{16, 0, 0, 0, 0, 0, 0, 0}).Equal(hashing0.Summary())
}
