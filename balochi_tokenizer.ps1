param(
    [string]$Mode = "TrainAndTest",
    [string]$TrainPath = "train_dataset.txt",
    [string]$TestPath = "test_dataset.txt",
    [string]$OutputDir = "model",
    [int]$VocabSize = 2000,
    [int]$MinFrequency = 2,
    [string]$CustomText = ""
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$csharpSource = @'
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;

public class BpePair : IEquatable<BpePair> {
    public string First;
    public string Second;

    public BpePair(string first, string second) {
        First = first;
        Second = second;
    }

    public bool Equals(BpePair other) {
        if (ReferenceEquals(null, other)) return false;
        if (ReferenceEquals(this, other)) return true;
        return First == other.First && Second == other.Second;
    }

    public override bool Equals(object obj) {
        return Equals(obj as BpePair);
    }

    public override int GetHashCode() {
        unchecked {
            return ((First != null ? First.GetHashCode() : 0) * 397) ^ (Second != null ? Second.GetHashCode() : 0);
        }
    }

    public override string ToString() {
        return First + " " + Second;
    }
}

public class BalochiBpeTokenizer {
    public Dictionary<string, int> TokenToId = new Dictionary<string, int>();
    public Dictionary<int, string> IdToToken = new Dictionary<int, string>();
    public Dictionary<BpePair, int> Merges = new Dictionary<BpePair, int>(); // pair -> rank
    public List<BpePair> MergeList = new List<BpePair>();

    public const string PAD_TOKEN = "<pad>";
    public const string UNK_TOKEN = "<unk>";
    public const string BOS_TOKEN = "<s>";
    public const string EOS_TOKEN = "</s>";
    public const string MASK_TOKEN = "<mask>";

    public int PadId { get { return TokenToId[PAD_TOKEN]; } }
    public int UnkId { get { return TokenToId[UNK_TOKEN]; } }
    public int BosId { get { return TokenToId[BOS_TOKEN]; } }
    public int EosId { get { return TokenToId[EOS_TOKEN]; } }
    public int MaskId { get { return TokenToId[MASK_TOKEN]; } }

    public BalochiBpeTokenizer() {
        InitSpecialTokens();
    }

    private void InitSpecialTokens() {
        TokenToId.Clear();
        IdToToken.Clear();
        Merges.Clear();
        MergeList.Clear();

        string[] specials = new string[] { BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, MASK_TOKEN };
        for (int i = 0; i < specials.Length; i++) {
            TokenToId[specials[i]] = i;
            IdToToken[i] = specials[i];
        }
    }

    public void Train(string trainFilePath, int targetVocabSize, int minFreq) {
        Console.WriteLine(string.Format("==> Starting BPE Tokenizer Training on: {0}", trainFilePath));
        DateTime startTime = DateTime.Now;

        InitSpecialTokens();

        // Step 1: Read all words and build frequency table
        Console.WriteLine("--> Reading training dataset...");
        Dictionary<string, int> wordFreqs = new Dictionary<string, int>();
        HashSet<char> alphabet = new HashSet<char>();

        using (StreamReader reader = new StreamReader(trainFilePath, Encoding.UTF8)) {
            string line;
            while ((line = reader.ReadLine()) != null) {
                if (string.IsNullOrEmpty(line)) continue;
                
                int count;
                if (wordFreqs.TryGetValue(line, out count)) {
                    wordFreqs[line] = count + 1;
                } else {
                    wordFreqs[line] = 1;
                }

                for (int i = 0; i < line.Length; i++) {
                    alphabet.Add(line[i]);
                }
            }
        }

        Console.WriteLine(string.Format("--> Unique training entries: {0}, Base alphabet characters: {1}", wordFreqs.Count, alphabet.Count));

        // Add base characters to vocab
        List<char> sortedAlphabet = alphabet.OrderBy(c => (int)c).ToList();
        foreach (char c in sortedAlphabet) {
            string s = c.ToString();
            if (!TokenToId.ContainsKey(s)) {
                int nextId = TokenToId.Count;
                TokenToId[s] = nextId;
                IdToToken[nextId] = s;
            }
        }

        Console.WriteLine(string.Format("--> Initial vocab size (special tokens + alphabet): {0}", TokenToId.Count));

        // Convert word frequencies to list of token sequences
        List<List<string>> splits = new List<List<string>>(wordFreqs.Count);
        List<int> freqs = new List<int>(wordFreqs.Count);

        foreach (KeyValuePair<string, int> kv in wordFreqs) {
            string word = kv.Key;
            List<string> charList = new List<string>(word.Length);
            for (int i = 0; i < word.Length; i++) {
                charList.Add(word[i].ToString());
            }
            splits.Add(charList);
            freqs.Add(kv.Value);
        }

        // BPE Merge Loop
        Console.WriteLine(string.Format("--> Learning BPE merges towards target vocab size: {0}...", targetVocabSize));
        int numMerges = targetVocabSize - TokenToId.Count;
        int mergesLearned = 0;

        while (TokenToId.Count < targetVocabSize) {
            // Count pair frequencies
            Dictionary<BpePair, int> pairCounts = new Dictionary<BpePair, int>();

            for (int w = 0; w < splits.Count; w++) {
                List<string> wordTokens = splits[w];
                int count = freqs[w];
                for (int i = 0; i < wordTokens.Count - 1; i++) {
                    BpePair pair = new BpePair(wordTokens[i], wordTokens[i + 1]);
                    int prev;
                    if (pairCounts.TryGetValue(pair, out prev)) {
                        pairCounts[pair] = prev + count;
                    } else {
                        pairCounts[pair] = count;
                    }
                }
            }

            if (pairCounts.Count == 0) {
                Console.WriteLine("--> No more pairs to merge.");
                break;
            }

            // Find best pair
            BpePair bestPair = null;
            int maxFreq = -1;

            foreach (KeyValuePair<BpePair, int> kv in pairCounts) {
                if (kv.Value > maxFreq) {
                    maxFreq = kv.Value;
                    bestPair = kv.Key;
                }
            }

            if (bestPair == null || maxFreq < minFreq) {
                Console.WriteLine(string.Format("--> Best pair frequency {0} below min frequency {1}. Stopping merges.", maxFreq, minFreq));
                break;
            }

            string mergedToken = bestPair.First + bestPair.Second;
            int rank = Merges.Count;
            Merges[bestPair] = rank;
            MergeList.Add(bestPair);

            if (!TokenToId.ContainsKey(mergedToken)) {
                int newId = TokenToId.Count;
                TokenToId[mergedToken] = newId;
                IdToToken[newId] = mergedToken;
            }

            mergesLearned++;

            if (mergesLearned % 200 == 0 || TokenToId.Count == targetVocabSize) {
                Console.WriteLine(string.Format("    [Merge {0}/{1}] Merged: '{2}' + '{3}' -> '{4}' (Freq: {5}, Vocab: {6})", 
                    mergesLearned, numMerges, bestPair.First, bestPair.Second, mergedToken, maxFreq, TokenToId.Count));
            }

            // Apply merge to all splits
            for (int w = 0; w < splits.Count; w++) {
                List<string> wordTokens = splits[w];
                if (wordTokens.Count < 2) continue;

                List<string> newWordTokens = new List<string>(wordTokens.Count);
                int i = 0;
                while (i < wordTokens.Count) {
                    if (i < wordTokens.Count - 1 && wordTokens[i] == bestPair.First && wordTokens[i + 1] == bestPair.Second) {
                        newWordTokens.Add(mergedToken);
                        i += 2;
                    } else {
                        newWordTokens.Add(wordTokens[i]);
                        i++;
                    }
                }
                splits[w] = newWordTokens;
            }
        }

        DateTime endTime = DateTime.Now;
        Console.WriteLine(string.Format("==> Training completed in {0:F2} seconds. Final Vocab Size: {1}, Total Merges: {2}", 
            (endTime - startTime).TotalSeconds, TokenToId.Count, Merges.Count));
    }

    public List<string> Tokenize(string text) {
        if (string.IsNullOrEmpty(text)) return new List<string>();

        // Start with characters
        List<string> tokens = new List<string>(text.Length);
        for (int i = 0; i < text.Length; i++) {
            tokens.Add(text[i].ToString());
        }

        if (tokens.Count < 2 || Merges.Count == 0) return tokens;

        while (true) {
            // Find lowest rank pair in tokens
            int minRank = int.MaxValue;
            BpePair bestPair = null;

            for (int i = 0; i < tokens.Count - 1; i++) {
                BpePair pair = new BpePair(tokens[i], tokens[i + 1]);
                int rank;
                if (Merges.TryGetValue(pair, out rank)) {
                    if (rank < minRank) {
                        minRank = rank;
                        bestPair = pair;
                    }
                }
            }

            if (bestPair == null) break;

            // Merge best pair
            List<string> newTokens = new List<string>(tokens.Count);
            int idx = 0;
            string merged = bestPair.First + bestPair.Second;
            while (idx < tokens.Count) {
                if (idx < tokens.Count - 1 && tokens[idx] == bestPair.First && tokens[idx + 1] == bestPair.Second) {
                    newTokens.Add(merged);
                    idx += 2;
                } else {
                    newTokens.Add(tokens[idx]);
                    idx++;
                }
            }
            tokens = newTokens;
            if (tokens.Count < 2) break;
        }

        return tokens;
    }

    public List<int> Encode(string text) {
        List<string> tokens = Tokenize(text);
        List<int> ids = new List<int>(tokens.Count);
        for (int i = 0; i < tokens.Count; i++) {
            int id;
            if (TokenToId.TryGetValue(tokens[i], out id)) {
                ids.Add(id);
            } else {
                ids.Add(UnkId);
            }
        }
        return ids;
    }

    public string Decode(List<int> ids) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ids.Count; i++) {
            int id = ids[i];
            string token;
            if (IdToToken.TryGetValue(id, out token)) {
                if (token != BOS_TOKEN && token != EOS_TOKEN && token != PAD_TOKEN && token != MASK_TOKEN) {
                    sb.Append(token);
                }
            } else {
                sb.Append(UNK_TOKEN);
            }
        }
        return sb.ToString();
    }

    public void Save(string outputDir) {
        if (!Directory.Exists(outputDir)) {
            Directory.CreateDirectory(outputDir);
        }

        Console.WriteLine(string.Format("==> Saving tokenizer model artifacts to: {0}", outputDir));

        // 1. Save vocab.json
        string vocabPath = Path.Combine(outputDir, "vocab.json");
        using (StreamWriter sw = new StreamWriter(vocabPath, false, Encoding.UTF8)) {
            sw.WriteLine("{");
            int count = 0;
            foreach (KeyValuePair<string, int> kv in TokenToId) {
                count++;
                string escapedKey = EscapeJsonString(kv.Key);
                string comma = (count < TokenToId.Count) ? "," : "";
                sw.WriteLine(string.Format("  \"{0}\": {1}{2}", escapedKey, kv.Value, comma));
            }
            sw.WriteLine("}");
        }

        // 2. Save merges.json
        string mergesJsonPath = Path.Combine(outputDir, "merges.json");
        using (StreamWriter sw = new StreamWriter(mergesJsonPath, false, Encoding.UTF8)) {
            sw.WriteLine("[");
            for (int i = 0; i < MergeList.Count; i++) {
                string comma = (i < MergeList.Count - 1) ? "," : "";
                string f = EscapeJsonString(MergeList[i].First);
                string s = EscapeJsonString(MergeList[i].Second);
                sw.WriteLine(string.Format("  [\"{0}\", \"{1}\"]{2}", f, s, comma));
            }
            sw.WriteLine("]");
        }

        // 3. Save merges.txt
        string mergesPath = Path.Combine(outputDir, "merges.txt");
        using (StreamWriter sw = new StreamWriter(mergesPath, false, Encoding.UTF8)) {
            sw.WriteLine("#version: 0.2");
            for (int i = 0; i < MergeList.Count; i++) {
                sw.WriteLine(MergeList[i].First + " " + MergeList[i].Second);
            }
        }

        // 4. Save tokenizer_config.json
        string configPath = Path.Combine(outputDir, "tokenizer_config.json");
        using (StreamWriter sw = new StreamWriter(configPath, false, Encoding.UTF8)) {
            sw.WriteLine("{");
            sw.WriteLine("  \"model_type\": \"bpe\",");
            sw.WriteLine("  \"language\": \"balochi\",");
            sw.WriteLine(string.Format("  \"vocab_size\": {0},", TokenToId.Count));
            sw.WriteLine(string.Format("  \"num_merges\": {0},", MergeList.Count));
            sw.WriteLine("  \"bos_token\": \"<s>\",");
            sw.WriteLine("  \"eos_token\": \"</s>\",");
            sw.WriteLine("  \"unk_token\": \"<unk>\",");
            sw.WriteLine("  \"pad_token\": \"<pad>\",");
            sw.WriteLine("  \"mask_token\": \"<mask>\"");
            sw.WriteLine("}");
        }

        // 5. Save tokenizer.json (HuggingFace format)
        string tokenizerJsonPath = Path.Combine(outputDir, "tokenizer.json");
        using (StreamWriter sw = new StreamWriter(tokenizerJsonPath, false, Encoding.UTF8)) {
            sw.WriteLine("{");
            sw.WriteLine("  \"version\": \"1.0\",");
            sw.WriteLine("  \"truncation\": null,");
            sw.WriteLine("  \"padding\": null,");
            sw.WriteLine("  \"added_tokens\": [");
            sw.WriteLine("    {\"id\": 0, \"content\": \"<s>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},");
            sw.WriteLine("    {\"id\": 1, \"content\": \"</s>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},");
            sw.WriteLine("    {\"id\": 2, \"content\": \"<pad>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},");
            sw.WriteLine("    {\"id\": 3, \"content\": \"<unk>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},");
            sw.WriteLine("    {\"id\": 4, \"content\": \"<mask>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}");
            sw.WriteLine("  ],");
            sw.WriteLine("  \"model\": {");
            sw.WriteLine("    \"type\": \"BPE\",");
            sw.WriteLine("    \"dropout\": null,");
            sw.WriteLine("    \"unk_token\": \"<unk>\",");
            sw.WriteLine("    \"continuing_subword_prefix\": null,");
            sw.WriteLine("    \"end_of_word_suffix\": null,");
            sw.WriteLine("    \"fuse_unk\": false,");
            sw.WriteLine("    \"vocab\": {");
            int vCount = 0;
            foreach (KeyValuePair<string, int> kv in TokenToId) {
                vCount++;
                string comma = (vCount < TokenToId.Count) ? "," : "";
                sw.WriteLine(string.Format("      \"{0}\": {1}{2}", EscapeJsonString(kv.Key), kv.Value, comma));
            }
            sw.WriteLine("    },");
            sw.WriteLine("    \"merges\": [");
            for (int i = 0; i < MergeList.Count; i++) {
                string comma = (i < MergeList.Count - 1) ? "," : "";
                string mergeStr = EscapeJsonString(MergeList[i].First + " " + MergeList[i].Second);
                sw.WriteLine(string.Format("      \"{0}\"{1}", mergeStr, comma));
            }
            sw.WriteLine("    ]");
            sw.WriteLine("  }");
            sw.WriteLine("}");
        }

        Console.WriteLine("==> All artifacts saved successfully.");
    }

    private static string EscapeJsonString(string s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.Length; i++) {
            char c = s[i];
            if (c == '\\') sb.Append("\\\\");
            else if (c == '\"') sb.Append("\\\"");
            else if (c == '\n') sb.Append("\\n");
            else if (c == '\r') sb.Append("\\r");
            else if (c == '\t') sb.Append("\\t");
            else if (c < 32) sb.Append(string.Format("\\u{0:x4}", (int)c));
            else sb.Append(c);
        }
        return sb.ToString();
    }

    public static string UnescapeJson(string s) {
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < s.Length) {
            if (s[i] == '\\' && i + 1 < s.Length) {
                char next = s[i + 1];
                if (next == '\"') { sb.Append('\"'); i += 2; }
                else if (next == '\\') { sb.Append('\\'); i += 2; }
                else if (next == 'n') { sb.Append('\n'); i += 2; }
                else if (next == 'r') { sb.Append('\r'); i += 2; }
                else if (next == 't') { sb.Append('\t'); i += 2; }
                else if (next == 'u' && i + 5 < s.Length) {
                    string hex = s.Substring(i + 2, 4);
                    int code = Convert.ToInt32(hex, 16);
                    sb.Append((char)code);
                    i += 6;
                } else {
                    sb.Append(next);
                    i += 2;
                }
            } else {
                sb.Append(s[i]);
                i++;
            }
        }
        return sb.ToString();
    }

    public void Load(string modelDir) {
        InitSpecialTokens();
        string vocabPath = Path.Combine(modelDir, "vocab.json");
        string mergesJsonPath = Path.Combine(modelDir, "merges.json");
        string mergesPath = Path.Combine(modelDir, "merges.txt");

        if (!File.Exists(vocabPath)) {
            throw new FileNotFoundException(string.Format("vocab.json not found in {0}", modelDir));
        }

        // Robust parsing of vocab.json
        string jsonText = File.ReadAllText(vocabPath, Encoding.UTF8);
        TokenToId.Clear();
        IdToToken.Clear();

        int idx = 0;
        while (idx < jsonText.Length) {
            int keyStart = jsonText.IndexOf('\"', idx);
            if (keyStart < 0) break;

            int keyEnd = keyStart + 1;
            while (keyEnd < jsonText.Length) {
                if (jsonText[keyEnd] == '\"' && jsonText[keyEnd - 1] != '\\') break;
                keyEnd++;
            }
            if (keyEnd >= jsonText.Length) break;

            string rawKey = jsonText.Substring(keyStart + 1, keyEnd - keyStart - 1);
            string key = UnescapeJson(rawKey);

            int colon = jsonText.IndexOf(':', keyEnd + 1);
            if (colon < 0) break;

            int valStart = colon + 1;
            while (valStart < jsonText.Length && char.IsWhiteSpace(jsonText[valStart])) valStart++;
            int valEnd = valStart;
            while (valEnd < jsonText.Length && (char.IsDigit(jsonText[valEnd]) || jsonText[valEnd] == '-')) valEnd++;

            if (valEnd > valStart) {
                string valStr = jsonText.Substring(valStart, valEnd - valStart);
                int val;
                if (int.TryParse(valStr, out val)) {
                    TokenToId[key] = val;
                    IdToToken[val] = key;
                }
            }
            idx = valEnd;
        }

        // Load merges
        Merges.Clear();
        MergeList.Clear();

        if (File.Exists(mergesJsonPath)) {
            string mJson = File.ReadAllText(mergesJsonPath, Encoding.UTF8);
            int mIdx = 0;
            while (mIdx < mJson.Length) {
                int openBracket = mJson.IndexOf('[', mIdx);
                if (openBracket < 0) break;

                int firstQuote = mJson.IndexOf('\"', openBracket);
                if (firstQuote < 0) break;
                int secondQuote = firstQuote + 1;
                while (secondQuote < mJson.Length && (mJson[secondQuote] != '\"' || mJson[secondQuote - 1] == '\\')) secondQuote++;
                if (secondQuote >= mJson.Length) break;

                int thirdQuote = mJson.IndexOf('\"', secondQuote + 1);
                if (thirdQuote < 0) break;
                int fourthQuote = thirdQuote + 1;
                while (fourthQuote < mJson.Length && (mJson[fourthQuote] != '\"' || mJson[fourthQuote - 1] == '\\')) fourthQuote++;
                if (fourthQuote >= mJson.Length) break;

                string first = UnescapeJson(mJson.Substring(firstQuote + 1, secondQuote - firstQuote - 1));
                string second = UnescapeJson(mJson.Substring(thirdQuote + 1, fourthQuote - thirdQuote - 1));

                BpePair pair = new BpePair(first, second);
                int rank = Merges.Count;
                Merges[pair] = rank;
                MergeList.Add(pair);

                mIdx = fourthQuote + 1;
            }
        } else if (File.Exists(mergesPath)) {
            string[] mLines = File.ReadAllLines(mergesPath, Encoding.UTF8);
            for (int i = 0; i < mLines.Length; i++) {
                string line = mLines[i];
                if (string.IsNullOrEmpty(line) || line.StartsWith("#")) continue;
                int spaceIdx = line.IndexOf(' ');
                if (spaceIdx > 0) {
                    string first = line.Substring(0, spaceIdx);
                    string second = line.Substring(spaceIdx + 1);
                    BpePair pair = new BpePair(first, second);
                    int rank = Merges.Count;
                    Merges[pair] = rank;
                    MergeList.Add(pair);
                }
            }
        }

        Console.WriteLine(string.Format("==> Loaded tokenizer model from '{0}': Vocab Size = {1}, Merges = {2}", modelDir, TokenToId.Count, Merges.Count));
    }

    public void Evaluate(string testFilePath) {
        Console.WriteLine("\n========================================================");
        Console.WriteLine(string.Format("==> Starting Evaluation on Test Corpus: {0}", testFilePath));
        Console.WriteLine("========================================================");

        string[] testLines = File.ReadAllLines(testFilePath, Encoding.UTF8);
        long totalOriginalChars = 0;
        long totalTokens = 0;
        int exactReconstructionCount = 0;
        int unkTokenCount = 0;
        int sampleDisplayCount = 15;

        Console.WriteLine(string.Format("Total test lines: {0}", testLines.Length));
        Console.WriteLine(string.Format("\n--- Sample Tokenizations (First {0} lines) ---", sampleDisplayCount));

        for (int i = 0; i < testLines.Length; i++) {
            string line = testLines[i];
            totalOriginalChars += line.Length;

            List<string> tokens = Tokenize(line);
            List<int> ids = Encode(line);
            totalTokens += ids.Count;

            for (int k = 0; k < ids.Count; k++) {
                if (ids[k] == UnkId) unkTokenCount++;
            }

            string reconstructed = Decode(ids);
            if (reconstructed == line) {
                exactReconstructionCount++;
            }

            if (i < sampleDisplayCount) {
                string tokenDisplay = string.Join(" | ", tokens.ToArray());
                string idsDisplay = string.Join(", ", ids.Select(x => x.ToString()).ToArray());
                Console.WriteLine(string.Format("[Line {0,2}] Text: {1}", i + 1, line));
                Console.WriteLine(string.Format("          Tokens ({0}): [ {1} ]", tokens.Count, tokenDisplay));
                Console.WriteLine(string.Format("          IDs: [ {0} ]", idsDisplay));
                Console.WriteLine(string.Format("          Reconstructed: {0} ({1})\n", reconstructed, (reconstructed == line ? "MATCH OK" : "MISMATCH FAIL")));
            }
        }

        double compressionRatio = (double)totalOriginalChars / (totalTokens > 0 ? totalTokens : 1);
        double fertilityRate = (double)totalTokens / (testLines.Length > 0 ? testLines.Length : 1);
        double accuracyPct = (double)exactReconstructionCount / (testLines.Length > 0 ? testLines.Length : 1) * 100.0;
        double unkPct = (double)unkTokenCount / (totalTokens > 0 ? totalTokens : 1) * 100.0;

        Console.WriteLine("---------------- Evaluation Summary ----------------");
        Console.WriteLine(string.Format("Total Test Entries       : {0}", testLines.Length));
        Console.WriteLine(string.Format("Total Original Chars     : {0}", totalOriginalChars));
        Console.WriteLine(string.Format("Total Tokens Produced    : {0}", totalTokens));
        Console.WriteLine(string.Format("Exact Reconstruction     : {0} / {1} ({2:F2}%)", exactReconstructionCount, testLines.Length, accuracyPct));
        Console.WriteLine(string.Format("Compression Ratio        : {0:F3} chars/token", compressionRatio));
        Console.WriteLine(string.Format("Fertility Rate           : {0:F3} tokens/word", fertilityRate));
        Console.WriteLine(string.Format("Unknown Tokens (<unk>)   : {0} ({1:F4}%)", unkTokenCount, unkPct));
        Console.WriteLine("========================================================\n");
    }
}
'@

Add-Type -TypeDefinition $csharpSource

$tokenizer = New-Object BalochiBpeTokenizer

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Balochi BPE Tokenizer Engine                " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$fullOutputDir = Join-Path (Get-Location) $OutputDir

if ($CustomText -ne "") {
    if (Test-Path (Join-Path $fullOutputDir "vocab.json")) {
        $tokenizer.Load($fullOutputDir)
    } else {
        $fullTrainPath = (Resolve-Path $TrainPath).Path
        $tokenizer.Train($fullTrainPath, $VocabSize, $MinFrequency)
        $tokenizer.Save($fullOutputDir)
    }

    $tokens = $tokenizer.Tokenize($CustomText)
    $ids = $tokenizer.Encode($CustomText)
    $reconstructed = $tokenizer.Decode($ids)

    Write-Host "`nInput Text   : $CustomText" -ForegroundColor Yellow
    Write-Host "Tokens ($($tokens.Count)): [ $($tokens -join ' | ') ]" -ForegroundColor Green
    Write-Host "Token IDs    : [ $($ids -join ', ') ]" -ForegroundColor Green
    Write-Host "Reconstructed: $reconstructed" -ForegroundColor Cyan
    return
}

if ($Mode -eq "TestOnly") {
    $tokenizer.Load($fullOutputDir)
    $fullTestPath = (Resolve-Path $TestPath).Path
    $tokenizer.Evaluate($fullTestPath)
    return
}

# 1. Train
$fullTrainPath = (Resolve-Path $TrainPath).Path
$tokenizer.Train($fullTrainPath, $VocabSize, $MinFrequency)

# 2. Save
$tokenizer.Save($fullOutputDir)

# 3. Test & Evaluate
if ($Mode -ne "TrainOnly") {
    $fullTestPath = (Resolve-Path $TestPath).Path
    $tokenizer.Evaluate($fullTestPath)
}

Write-Host "Processing complete! All artifacts generated in: $fullOutputDir" -ForegroundColor Green
