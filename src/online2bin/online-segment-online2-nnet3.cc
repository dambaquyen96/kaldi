// online2bin/online-segment-online2-nnet3-live.cc

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

#include "base/kaldi-common.h"
#include "base/timer.h"
#include "decoder/faster-decoder.h"
#include "feat/feature-functions.h"
#include "feat/feature-mfcc.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "hmm/hmm-utils.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/sausages.h"
#include "matrix/kaldi-matrix.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "transform/cmvn.h"
#include "util/kaldi-thread.h"
#include "util/common-utils.h"
#include <stdio.h>
#include <stdlib.h>

#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <hiredis/hiredis.h>
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

bool AccCmvnStatsWrapper(std::string utt,
                         const MatrixBase<BaseFloat> &feats,
                         RandomAccessBaseFloatVectorReader *weights_reader,
                         Matrix<double> *cmvn_stats) {
  if (!weights_reader->IsOpen()) {
    AccCmvnStats(feats, NULL, cmvn_stats);
    return true;
  } else {
    if (!weights_reader->HasKey(utt)) {
      KALDI_WARN << "No weights available for utterance " << utt;
      return false;
    }
    const Vector<BaseFloat> &weights = weights_reader->Value(utt);
    if (weights.Dim() != feats.NumRows()) {
      KALDI_WARN << "Weights for utterance " << utt << " have wrong dimension "
                 << weights.Dim() << " vs. " << feats.NumRows();
      return false;
    }
    AccCmvnStats(feats, &weights, cmvn_stats);
    return true;
  }
}

fst::Fst<fst::StdArc> *ReadNetwork(std::string filename) {
  /* read decoding network FST */
  Input ki(filename); /* use ki.Stream() instead of is. */
  if (!ki.Stream().good()) KALDI_ERR << "Could not open decoding-graph FST "
                                      << filename;
  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), "<unknown>")) {
    KALDI_ERR << "Reading FST: error reading FST header.";
  }
  if (hdr.ArcType() != fst::StdArc::Type()) {
    KALDI_ERR << "FST with arc type " << hdr.ArcType() << " not supported.";
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = NULL;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else {
    KALDI_ERR << "Reading FST: unsupported FST type: " << hdr.FstType();
  }
  if (decode_fst == NULL) { /* fst code will warn. */
    KALDI_ERR << "Error reading FST (after reading header).";
    return NULL;
  } else {
    return decode_fst;
  }
}

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
    // std::cerr << "[" << utt << "] ";
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
      strResult += s;
      strResult += ' ';
    }
    std::cerr << std::endl;
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

bool AppendFeats(const std::vector<Matrix<BaseFloat> > &in,
                 std::string utt,
                 int32 tolerance,
                 Matrix<BaseFloat> *out) {
  // Check the lengths
  int32 min_len = in[0].NumRows(),
      max_len = in[0].NumRows(),
      tot_dim = in[0].NumCols();
  for (int32 i = 1; i < in.size(); i++) {
    int32 len = in[i].NumRows(), dim = in[i].NumCols();
    tot_dim += dim;
    if(len < min_len) min_len = len;
    if(len > max_len) max_len = len;
  }
  if (max_len - min_len > tolerance || min_len == 0) {
    KALDI_WARN << "Length mismatch " << max_len << " vs. " << min_len
               << (utt.empty() ? "" : " for utt ") << utt
               << " exceeds tolerance " << tolerance;
    out->Resize(0, 0);
    return false;
  }
  if (max_len - min_len > 0) {
    KALDI_VLOG(2) << "Length mismatch " << max_len << " vs. " << min_len
                  << (utt.empty() ? "" : " for utt ") << utt
                  << " within tolerance " << tolerance;
  }
  out->Resize(min_len, tot_dim);
  int32 dim_offset = 0;
  for (int32 i = 0; i < in.size(); i++) {
    int32 this_dim = in[i].NumCols();
    out->Range(0, min_len, dim_offset, this_dim).CopyFromMat(
        in[i].Range(0, min_len, 0, this_dim));
    dim_offset += this_dim;
  }
  return true;
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
    std::vector<std::string> v_start;
    std::vector<std::string> v_end;
    std::vector<std::string> v_sentence;
	  std::vector<BaseFloat> v_confidence;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Read utt and path from REDIS queue, decode online with nnet3.\n"
        "Result is set in REDIS with variable name is <utt>.\n"
        "Also support REDIS cluster.\n"
        "\n"
        "Usage: online-segment-online2-nnet3 [options] <nnet3-in> <fst-in> <host:port> <queue>\n";

    ParseOptions po(usage);
    bool s_allow_partial = true;
    BaseFloat s_acoustic_scale = 0.1;
    BaseFloat s_vtln_warp = 1.0;    
    int32 s_channel = -1;
    int32 s_left_context = 4, s_right_context = 4;
    int32 s_length_tolerance = 2;
    MfccOptions s_mfcc_opts;
    FasterDecoderOptions s_decoder_opts;
    PitchExtractionOptions s_pitch_opts;
    ProcessPitchOptions s_process_opts;

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

    po.Register("s-max-active", &s_decoder_opts.max_active,
                "Segment max active");
    po.Register("s-beam", &s_decoder_opts.beam,
                "Segment beam");
    po.Register("s-sample-frequency", &s_mfcc_opts.frame_opts.samp_freq,
                "Segment sample frequency");
    po.Register("s-pitch-sample-frequency", &s_pitch_opts.samp_freq,
                "Segment sample frequency");
    po.Register("s-acoustic-scale", &s_acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("s-allow-partial", &s_allow_partial,
                "Produce output even when final state was not reached");
    po.Register("s-use-energy", &s_mfcc_opts.use_energy,
                "Segment use energy");
    po.Register("s-num-mel-bins", &s_mfcc_opts.mel_opts.num_bins,
                "Segment num mel bins");
    po.Register("s-num-ceps", &s_mfcc_opts.num_ceps,
                "Segment num ceps");
    po.Register("s-low-freq", &s_mfcc_opts.mel_opts.low_freq,
                "Segment low frequency");
    po.Register("s-high-freq", &s_mfcc_opts.mel_opts.high_freq,
                "Segment high frequency");
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

    if (po.NumArgs() != 14) {
      po.PrintUsage();
      return 1;
    }

    std::string s_model_rspecifier = po.GetArg(1),
        s_fst_rspecifier = po.GetArg(2),
        s_transform_rspecifier_or_rxfilename = po.GetOptArg(3),
        s_word_syms_filename = po.GetArg(4),
        nnet3_rxfilename = po.GetArg(5),
        fst_rxfilename = po.GetArg(6),
        str_host = po.GetArg(7),
        str_queue = po.GetArg(8),
        str_decode_queue = po.GetArg(9),
        pred_file = po.GetArg(10),
        str_output_queue = po.GetArg(11),
        str_commands = po.GetArg(12),
        str_precommands = po.GetArg(13),
		    str_sufcommands = po.GetArg(14);

    hostname = str_host.c_str();
    queue = str_queue.c_str();

    /*================== SEGMENT ===================*/
    /* Init decoder */
    TransitionModel s_trans_model;
    AmDiagGmm s_am_gmm;
    {
      bool s_binary;
      Input s_ki(s_model_rspecifier, &s_binary);
      s_trans_model.Read(s_ki.Stream(), s_binary);
      s_am_gmm.Read(s_ki.Stream(), s_binary);
    }

    fst::SymbolTable *s_word_syms = NULL;
    if (!(s_word_syms = fst::SymbolTable::ReadText(s_word_syms_filename)))
      KALDI_ERR << "Could not read symbol table from file "
                  << s_word_syms_filename;

    fst::Fst<fst::StdArc> *s_decode_fst = ReadNetwork(s_fst_rspecifier);

    Mfcc s_mfcc(s_mfcc_opts);
    FasterDecoder s_decoder(*s_decode_fst, s_decoder_opts);
      
    /* Init transform lda */
    RandomAccessBaseFloatMatrixReaderMapped s_transform_reader;
    Matrix<BaseFloat> s_global_transform;
    ReadKaldiObject(s_transform_rspecifier_or_rxfilename, &s_global_transform);

    /*================== DECODE ==================*/
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

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    OnlineTimingStats timing_stats;

    // QuyenDB - Start
    printf("Cluster host with %s\n", hostname);
    redis::cluster::Cluster *cluster = new redis::cluster::Cluster();

    if( cluster->setup(hostname, true)!=0 ) {
        std::cerr << "Cluster setup fail" << std::endl;
        exit(1);
    }
    printf("Processing on %s! Waiting for client ...\n", queue);
    std::cout << "Start" << std::endl;
    while(1){
      try {
        commands.clear();
        commands.push_back("LLEN");
        commands.push_back(str_queue);
        reply = cluster->run(commands);
        if(reply->integer == 0){
          /* std::cout << "Not found" << std::endl; */
          freeReplyObject(reply);
          continue;
        }
        freeReplyObject(reply);
        commands.clear();
        commands.push_back("RPOP");
        commands.push_back(str_queue);
        reply = cluster->run(commands);
        if(reply->type == REDIS_REPLY_NIL){
          freeReplyObject(reply);
          continue;
        }
        std::string s_scp(reply->str);
        freeReplyObject(reply);

        std::istringstream s_iss(s_scp);
        int s_index = 0;
        while(s_iss) s_iss >> arr_str[s_index++];
        s_index--;
        if(s_index != 4){
          std::cout << "Found " << s_index << " token, expected format <utt> <path> <json> <output_folder> in: " << s_scp << std::endl;
          continue;
        }
        std::string utt = arr_str[0];
        std::string path = arr_str[1];
        std::string json_path = arr_str[2];
        std::string output_folder = arr_str[3];
        
        std::string s_utt = utt;
        std::string s_path = path;

        printf("Processing %s\n", s_utt.c_str());
        std::chrono::steady_clock::time_point s_begin = std::chrono::steady_clock::now();
        // std::string precmd = str_precommands + " " + nmf_path + " " + path;
        // std::cout << "Execute: " << precmd << std::endl;
        // system(precmd.c_str());

        std::string wav_rspecifier = "scp:echo " + s_utt + " " + s_path + "|";
        
        RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
        if(!wav_reader.HasKey(s_utt)){
          continue;
        }
        const WaveData &wav_data = wav_reader.Value(s_utt);
        SubVector<BaseFloat> data(wav_data.Data(), 0);

        int32 s_num_chan = wav_data.Data().NumRows(), s_this_chan = s_channel;
        {  // This block works out the channel (0=left, 1=right...)
          KALDI_ASSERT(s_num_chan > 0);  // should have been caught in
          // reading code if no channels.
          if (s_channel == -1) {
            s_this_chan = 0;
            if (s_num_chan != 1)
              KALDI_WARN  << "Channel not specified but you have data with "
                    << s_num_chan  << " channels; defaulting to zero";
          } else {
            if (s_this_chan >= s_num_chan) {
              KALDI_WARN  << "File with id " << s_utt << " has "
                    << s_num_chan << " channels but you specified channel "
                    << s_channel << ", producing no output.";
              continue;
            }
          }
        }

        /* Compute raw mfcc */
        BaseFloat s_vtln_warp_local;
        s_vtln_warp_local = s_vtln_warp;
        SubVector<BaseFloat> s_waveform(wav_data.Data(), s_this_chan);
        Matrix<BaseFloat> s_raw, s_pitch, s_processed_pitch, s_features;
        s_mfcc.ComputeFeatures(s_waveform, wav_data.SampFreq(), s_vtln_warp_local, &s_raw);
        ComputeKaldiPitch(s_pitch_opts, s_waveform, &s_pitch);
        ProcessPitch(s_process_opts, s_pitch, &s_processed_pitch);
        vector<Matrix<BaseFloat> > arr_feats(2);
        arr_feats[0] = s_raw;
        arr_feats[1] = s_processed_pitch;
        AppendFeats(arr_feats, s_utt, s_length_tolerance, &s_features);

        /* Compute cmvn */
        std::string s_weights_rspecifier;
        RandomAccessBaseFloatVectorReader s_weights_reader(s_weights_rspecifier);
        Matrix<double> s_stats;
        InitCmvnStats(s_features.NumCols(), &s_stats);
        AccCmvnStatsWrapper(s_utt, s_features, &s_weights_reader, &s_stats);  
        
        /* Apply cmvn */
        ApplyCmvn(s_stats, false, &s_features);
        
        /* Splice feats */
        Matrix<BaseFloat> s_spliced;
        SpliceFrames(s_features, s_left_context, s_right_context, &s_spliced);
        /* Transform feats */
        const Matrix<BaseFloat> &s_trans = s_global_transform;
        int32   s_transform_rows = s_trans.NumRows(),
                s_transform_cols = s_trans.NumCols(),
                s_feat_dim = s_spliced.NumCols();
                
        Matrix<BaseFloat> s_feat_out(s_spliced.NumRows(), s_transform_rows);
        if (s_transform_cols == s_feat_dim) {
          s_feat_out.AddMatMat(1.0, s_spliced, kNoTrans, s_trans, kTrans, 0.0);
        } else if (s_transform_cols == s_feat_dim + 1) {
          SubMatrix<BaseFloat> s_linear_part(s_trans, 0, s_transform_rows, 0, s_feat_dim);
          s_feat_out.AddMatMat(1.0, s_spliced, kNoTrans, s_linear_part, kTrans, 0.0);
          Vector<BaseFloat> s_offset(s_transform_rows);
          s_offset.CopyColFromMat(s_trans, s_feat_dim);
          s_feat_out.AddVecToRows(1.0, s_offset);
        } else {
          KALDI_WARN << "Transform matrix for utterance " << s_utt << " has bad dimension "
                << s_transform_rows << "x" << s_transform_cols << " versus feat dim "
                << s_feat_dim;
          if (s_transform_cols == s_feat_dim+2)
            KALDI_WARN << "[perhaps the transform was created by compose-transforms, "
                    "and you forgot the --b-is-affine option?]";
          continue;
        }
              
        s_features = s_feat_out;

        DecodableAmDiagGmmScaled s_gmm_decodable(s_am_gmm, s_trans_model, s_features,
                                           s_acoustic_scale);
        s_decoder.Decode(&s_gmm_decodable);
        fst::VectorFst<LatticeArc> s_decoded;
        if ( (s_allow_partial || s_decoder.ReachedFinal()) && s_decoder.GetBestPath(&s_decoded) ) {
          std::vector<int32> s_alignment;
          std::vector<int32> s_words;
          LatticeWeight s_weight;
          GetLinearSymbolSequence(s_decoded, &s_alignment, &s_words, &s_weight);
        
          if (s_word_syms != NULL) {
            for (size_t i = 0; i < s_words.size(); i++) {
              std::string s = s_word_syms->Find(s_words[i]);
              if (s == "")
                KALDI_ERR << "Word-id " << s_words[i] <<" not in symbol table.";
              std::cerr << s << ' ';
            }
            std::cerr << '\n';
          }
          
          /* Write pred */
          Int32VectorWriter s_phones_writer("ark,t:" + pred_file + ".int");
          std::vector<std::vector<int32> > s_split;
          SplitToPhones(s_trans_model, s_alignment, &s_split);
          std::vector<int32> s_phones;
          for (size_t i = 0; i < s_split.size(); i++) {
            KALDI_ASSERT(!s_split[i].empty());
            int32 s_phone = s_trans_model.TransitionIdToPhone(s_split[i][0]);
            int32 s_num_repeats = s_split[i].size();
            for(int32 j = 0; j < s_num_repeats; j++){
              s_phones.push_back(s_phone);
            }
          }
          s_phones_writer.Write(s_utt, s_phones);
        } else {
          KALDI_WARN  << "Did not successfully decode utterance " << s_utt
                << ", len = " << s_features.NumRows();
        }

        std::string cmd = str_commands + " " + pred_file + " "  + str_decode_queue;
		    std::cout << "Execute: " << cmd << std::endl;
        system(cmd.c_str());

        std::chrono::steady_clock::time_point s_end = std::chrono::steady_clock::now();
        double s_elapsed_secs = double(std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_begin).count()) / 1000;
        std::cout << "Segmentation finish in " << s_elapsed_secs << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        while(1){
          commands.clear();
          commands.push_back("LLEN");
          commands.push_back(str_decode_queue);
          reply = cluster->run(commands);
          if(reply->integer == 0){
            freeReplyObject(reply);
            continue;
          }
          freeReplyObject(reply);

          commands.clear();
          commands.push_back("RPOP");
          commands.push_back(str_decode_queue);
          reply = cluster->run(commands);
          std::string scp(reply->str);
          freeReplyObject(reply);

          std::istringstream iss(scp);
          int index = 0;
          while(iss) iss >> arr_str[index++];
          index--;
          if(index == 1){
            std::cout << "[PACKING] Utt:" << utt << std::endl;
            std::cout << "[PACKING] Num of segments: " << v_sentence.size() << std::endl;
      			std::string utt_output_folder = output_folder + "/" + utt;
      			
      			std::string mkdir_cmd = "mkdir -p " + utt_output_folder + "; " + "rm -f " + utt_output_folder + "/*";
      			system(mkdir_cmd.c_str());
                  
      			std::string output_file = utt_output_folder + "/" + utt + ".txt";
            std::ofstream file_writer(output_file);
            if (file_writer.is_open()){
              for(int i = 0; i < v_sentence.size(); i++){
                std::string line = "";
                if (get_confidence){
                  std::stringstream conf_buf;
				  conf_buf << v_confidence[i];
                  line = utt + " " + v_start[i] + " " + v_end[i] + " " + conf_buf.str() + " " + v_sentence[i] + "\n";
                } else {
                  line = utt + " " + v_start[i] + " " + v_end[i] + " " + v_sentence[i] + "\n";
                }
                file_writer << line;
              }
              std::cout << "[INFO] Write to: " << output_file << std::endl;
              file_writer.close();
              //commands.clear();
              //commands.push_back("LPUSH");
              //commands.push_back(str_output_queue);
              //commands.push_back(utt + " " + path + " " + output_file);
              //reply = cluster->run(commands);
              //freeReplyObject(reply);
            }
            else {
              std::cout << "[ERROR] Unable to open " << output_file << std::endl;
            }

            std::string path_file = utt_output_folder + "/" + utt + ".path";
            std::ofstream file_writer2(path_file);
            if (file_writer2.is_open()){
              file_writer2 << path << std::endl;
              file_writer2.close();
            }
            else {
              std::cout << "[ERROR] Unable to open " << path_file << std::endl;
            }

      			std::string sufcmd = str_sufcommands + " " + utt + " " + path + " " + output_file + " " + utt_output_folder + " " + json_path;
      			std::cout << "Execute: " << sufcmd << std::endl;
            system(sufcmd.c_str());
      			std::cout << "[INFO] Output folder: " << utt_output_folder << std::endl;

            v_start.clear();
            v_end.clear();
            v_sentence.clear();
            v_confidence.clear();

            std::chrono::steady_clock::time_point final_end = std::chrono::steady_clock::now();
            double final_elapsed_secs = double(std::chrono::duration_cast<std::chrono::milliseconds>(final_end - s_begin).count()) / 1000;
            std::cout << "[INFO] Processing in " << final_elapsed_secs << std::endl;
            std::cout << "=========================================" << std::endl;
            
            break;
          }

          if(index != 2){
            std::cout << "Found " << index << " token, expected format <start> <end> in: " << scp << std::endl;
            break;
          }
          std::string seg_st = arr_str[0];
          std::string seg_ed = arr_str[1];
          v_start.push_back(seg_st);
          v_end.push_back(seg_ed);

          std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
          
          std::string wav_rspecifier = "ark:extract-segments-single "
                                      + utt + " " + path + " "
                                      + seg_st + " " + seg_ed + " "
                                      + "ark:- |";


          OnlineIvectorExtractorAdaptationState adaptation_state(
              feature_info.ivector_extractor_info);
          RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
          if(!wav_reader.HasKey(utt)){
            continue;
          }
          const WaveData &wave_data = wav_reader.Value(utt);
          SubVector<BaseFloat> data(wave_data.Data(), 0);
        
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

          int ok = 1;
          while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                           : samp_remaining;

            SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
            try {
              feature_pipeline.AcceptWaveform(samp_freq, wave_part);
            } catch (...) {
              ok = 0;
              break;
            }

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

            if (do_endpointing && decoder.EndpointDetected(endpoint_opts))
              break;
          }
          if(ok == 0) continue;
          decoder.FinalizeDecoding();

          CompactLattice clat;
          bool end_of_utterance = true;
          decoder.GetLattice(end_of_utterance, &clat);

          std::string transcript = GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                       &num_frames, &tot_like);
          v_sentence.push_back(transcript);
          if (get_confidence) 
            v_confidence.push_back(GetConfidenceScore(clat, decodable_opts.acoustic_scale, 1.0));
          num_done++;
          std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
          double elapsed_secs = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) / 1000;
          std::cout << "Decode in " << elapsed_secs << std::endl;
          std::cout << "------------------------------------" << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << e.what();
        continue;
      }
    }

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
