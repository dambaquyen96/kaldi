// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

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

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

#include "lat/sausages.h"

#include <arpa/inet.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <hiredis.h>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <redis_cluster.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kaldi {

std::string GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  std::string strResult = "";
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return "";
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  // KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
  //               << (likelihood / num_frames) << " over " << num_frames
  //               << " frames.";

  if (word_syms != NULL) {
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      strResult += s;
      strResult += ' ';
    }
  }
  return strResult;
}

BaseFloat GetConfidenceScore(CompactLattice clat, BaseFloat acoustic_scale, BaseFloat lm_scale){
	fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
	MinimumBayesRisk *mbr = NULL;
	MinimumBayesRiskOptions mbr_opts;
	mbr = new MinimumBayesRisk(clat, mbr_opts);
	const std::vector<BaseFloat> &conf = mbr->GetOneBestConfidences();
	BaseFloat conf_sentence = 0.0;
	for (size_t i = 0; i < conf.size(); i++) {
		conf_sentence += conf[i];
	}
	conf_sentence /= conf.size();
	return conf_sentence;
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    std::string arr_str[10];
    redisReply *reply;
    std::vector<std::string> commands;
    const char *hostname = "127.0.0.1:6379";
    const char *queue = "decode_jobs";

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage = "";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 0.18;
    bool do_endpointing = false;
    bool online = true;
    bool get_confidence = false;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");
    po.Register("get-confidence", &get_confidence,
                "If true, get sentence-level confidence score.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);


    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        str_host = po.GetArg(3),
        str_queue = po.GetArg(4);

    hostname = str_host.c_str();
    queue = str_queue.c_str();

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);


    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);
    std::cout << "Model loaded" << std::endl;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    OnlineTimingStats timing_stats;

    std::cout << "Cluster host with " << hostname << std::endl;
    redis::cluster::Cluster *cluster = new redis::cluster::Cluster();

    if( cluster->setup(hostname, true)!=0 ) {
        std::cerr << "Cluster setup fail" << std::endl;
        exit(1);
    }

    while(1) {
      try {
        commands.clear();
        commands.push_back("LLEN");
        commands.push_back(str_queue);
        reply = cluster->run(commands);
        if(reply->integer == 0){
          freeReplyObject(reply);
          std::this_thread::sleep_for(std::chrono::microseconds(10));
          continue;
        }
        freeReplyObject(reply);
        commands.clear();
        commands.push_back("RPOP");
        commands.push_back(str_queue);
        reply = cluster->run(commands);
        if(reply->type == REDIS_REPLY_NIL){
          freeReplyObject(reply);
          std::this_thread::sleep_for(std::chrono::microseconds(10));
          continue;
        }
        std::string s_scp(reply->str);
        freeReplyObject(reply);

        std::istringstream s_iss(s_scp);
        int s_index = 0;
        while(s_iss) s_iss >> arr_str[s_index++];
        s_index--;
        if(s_index != 2){
          std::cout << "Found " << s_index << " token, expected format <utt> <path> in: " << s_scp << std::endl;
          continue;
        }
        std::string utt = arr_str[0];
        std::string path = arr_str[1];

        std::cout << "Processing: " << utt << std::endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::string wav_rspecifier = "scp:echo " + utt + " " + path + "|";
        RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
        if(!wav_reader.HasKey(utt)){
          continue;
        }
        const WaveData &wave_data = wav_reader.Value(utt);
        SubVector<BaseFloat> data(wave_data.Data(), 0);
        BaseFloat wave_duration = wave_data.Duration();

        OnlineIvectorExtractorAdaptationState adaptation_state(
              feature_info.ivector_extractor_info);
        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);

        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);

        SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);
        OnlineTimer decoding_timer(utt);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(samp_freq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;

        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;
          decoding_timer.WaitUntil(samp_offset / samp_freq);
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }

          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
          }

          decoder.AdvanceDecoding();

          if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
            break;
          }
        }
        decoder.FinalizeDecoding();

        CompactLattice clat;
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat);

        std::string transcript = GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                       &num_frames, &tot_like);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double elapsed_secs = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) / 1000;
        double real_time_factor = elapsed_secs / double(wave_duration);

        commands.clear();
        commands.push_back("SET");
        commands.push_back(utt);
        commands.push_back(transcript);
        reply = cluster->run(commands);
        freeReplyObject(reply);

        std::cout << "Text: " << transcript << std::endl;
        std::cout << "Duration(s): " << wave_duration << " | Time(s): " << elapsed_secs << " | RTF: " << real_time_factor << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

      } catch (const std::exception& e) {
        std::cerr << e.what();
        continue;
      }
    }

    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
