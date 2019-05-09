// featbin/extract-segments.cc

// Copyright 2009-2011  Microsoft Corporation;  Govivace Inc.
//           2013       Arnab Ghoshal

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

/*! @brief This is the main program for extracting segments from a wav file
 - usage : 
     - extract-segments [options ..]  <scriptfile > <segments-file> <wav-written-specifier>
     - "scriptfile" must contain full path of the wav file.
     - "segments-file" should have the information of the segments that needs to be extracted from wav file
     - the format of the segments file : speaker_name wavfilename start_time(in secs) end_time(in secs) channel-id(0 or 1)
     - The channel-id is 0 for the left channel and 1 for the right channel.  This is not required for mono recordings.
     - "wav-written-specifier" is the output segment format
*/
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    
    const char *usage =
        "Extract segments from a large audio file in WAV format.\n"
		"Usage: extract-segments-single <utt> <path> <start-time> <endtime> <wav-wspecifier>"
        "e.g. extract-segments utt dir/tmp.wav 0.23 0.1 ark:- | <some-other-program>\n";

    ParseOptions po(usage);
    BaseFloat min_segment_length = 0.1, // Minimum segment length in seconds.
        max_overshoot = 0.5;  // max time by which last segment can overshoot
    po.Register("min-segment-length", &min_segment_length,
                "Minimum segment length in seconds (reject shorter segments)");
    po.Register("max-overshoot", &max_overshoot,
                "End segments overshooting audio by less than this (in seconds) "
                "are truncated, else rejected.");
    
    po.Read(argc, argv);
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

      std::string utt = po.GetArg(1),
        path = po.GetArg(2),
				start_str = po.GetArg(3),
				end_str = po.GetArg(4),
				wav_wspecifier = po.GetArg(5);

      std::string wav_rspecifier = "scp:echo " + utt + " " + path + "|";

      RandomAccessTableReader<WaveHolder> reader(wav_rspecifier);
      TableWriter<WaveHolder> writer(wav_wspecifier);
      // Convert the start time and endtime to real from string. Segment is
      // ignored if start or end time cannot be converted to real.
      double start, end;
      if (!ConvertStringToReal(start_str, &start)) {
        KALDI_WARN << "Invalid argument [bad start]";
        return -1;
      }
      if (!ConvertStringToReal(end_str, &end)) {
        KALDI_WARN << "Invalid argument [bad end]";
        return -1;
      }
      // start time must not be negative; start time must not be greater than
      // end time, except if end time is -1
      if (start < 0 || (end != -1.0 && end <= 0) || ((start >= end) && (end > 0))) {
        KALDI_WARN << "Invalid argument [empty or invalid segment]";
        return -1;
      }
      
      const WaveData &wave = reader.Value(utt);
      const Matrix<BaseFloat> &wave_data = wave.Data();
      BaseFloat samp_freq = wave.SampFreq();  // read sampling fequency
      int32 num_samp = wave_data.NumCols();  // number of samples in recording

      // Convert starting time of the segment to corresponding sample number.
      // If end time is -1 then use the whole file starting from start time.
      int32 start_samp = start * samp_freq,
          end_samp = (end != -1)? (end * samp_freq) : num_samp;
      KALDI_ASSERT(start_samp >= 0 && end_samp > 0 && "Invalid start or end.");

      // start sample must be less than total number of samples,
      // otherwise skip the segment
      if (start_samp < 0 || start_samp >= num_samp) {
        KALDI_WARN << "Start sample out of range " << start_samp << " [length:] "
                   << num_samp << ", skipping file";
        return -1;
      }
      /* end sample must be less than total number samples 
       * otherwise skip the segment
       */
      if (end_samp > num_samp) {
        if ((end_samp >=
             num_samp + static_cast<int32>(max_overshoot * samp_freq))) {
          KALDI_WARN << "End sample too far out of range " << end_samp
                     << " [length:] " << num_samp << ", skipping file";
          return -1;
        }
        end_samp = num_samp;  // for small differences, just truncate.
      }
      // Skip if segment size is less than minimum segment length (default 0.1s)
      if (end_samp <=
          start_samp + static_cast<int32>(min_segment_length * samp_freq)) {
        KALDI_WARN << "File too short, skipping it.";
        return -1;
      }
      SubMatrix<BaseFloat> segment_matrix(wave_data, 0, 1, start_samp, end_samp-start_samp);
      WaveData segment_wave(samp_freq, segment_matrix);
      writer.Write(utt, segment_wave); // write segment in wave format.
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

