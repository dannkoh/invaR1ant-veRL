package main

import (
	"bufio"
	"crypto/md5"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/writer"
)

// SMT2File holds the parsed contents of an SMT2 file.
type SMT2File struct {
	Constants string
	Solution  string
}

// QuestionEntry is the structure we output in the Parquet file.
type QuestionEntry struct {
	Problem         string `parquet:"name=problem, type=BYTE_ARRAY, convertedtype=UTF8"`
	ExampleIndices  string `parquet:"name=example_indices, type=BYTE_ARRAY, convertedtype=UTF8"`
	Examples        string `parquet:"name=examples, type=BYTE_ARRAY, convertedtype=UTF8"`
	Question        string `parquet:"name=question, type=BYTE_ARRAY, convertedtype=UTF8"`
	AnswerIndex     int    `parquet:"name=answer_index, type=INT32"`
	AnswerConstants string `parquet:"name=answer_constants, type=BYTE_ARRAY, convertedtype=UTF8"`
	AnswerSolution  string `parquet:"name=answer_solution, type=BYTE_ARRAY, convertedtype=UTF8"`
	VariableMapping string `parquet:"name=variable_mapping, type=BYTE_ARRAY, convertedtype=UTF8"`
}

// IndexPair represents a combination (list of indices) and an answer index.
type IndexPair struct {
	Combo       []int
	AnswerIndex int
}

// parseSMT2File reads an SMT2 file and returns its constants and solution.
func parseSMT2File(path string) (SMT2File, error) {
	f, err := os.Open(path)
	if err != nil {
		return SMT2File{}, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var constLines, solLines []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "(declare-const") {
			constLines = append(constLines, line)
		} else if strings.HasPrefix(line, "(assert") {
			solLines = append(solLines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return SMT2File{}, err
	}
	return SMT2File{
		Constants: strings.Join(constLines, "\n"),
		Solution:  strings.Join(solLines, "\n"),
	}, nil
}

// groupSMT2Files walks the base directory and groups SMT2 file paths by problem name and index.
// Expected file name pattern: own.<problem>_<N>.smt2
func groupSMT2Files(baseDir string) (map[string]map[int]string, error) {
	problemsFiles := make(map[string]map[int]string)
	pattern := regexp.MustCompile(`^own\.(.+)_(\d+)\.smt2$`)
	err := filepath.Walk(baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if filepath.Ext(info.Name()) == ".smt2" {
			matches := pattern.FindStringSubmatch(info.Name())
			if len(matches) == 3 {
				problem := matches[1]
				idx, err := strconv.Atoi(matches[2])
				if err != nil {
					return err
				}
				if _, ok := problemsFiles[problem]; !ok {
					problemsFiles[problem] = make(map[int]string)
				}
				problemsFiles[problem][idx] = path
			}
		}
		return nil
	})
	return problemsFiles, err
}

// extractVariableNames returns unique variable names found in constantsText.
func extractVariableNames(constantsText string) []string {
	varNamesSet := make(map[string]struct{})
	pattern := regexp.MustCompile(`\(declare-const\s+(\S+)`)
	matches := pattern.FindAllStringSubmatch(constantsText, -1)
	for _, match := range matches {
		if len(match) > 1 {
			varNamesSet[match[1]] = struct{}{}
		}
	}
	var varNames []string
	for k := range varNamesSet {
		varNames = append(varNames, k)
	}
	sort.Strings(varNames)
	return varNames
}

// renameVariablesInTexts renames variables in each text (and in the constants) based on a random mapping.
func renameVariablesInTexts(texts []string, constantsText string, seed int64) ([]string, string, map[string]string) {
	mapping := make(map[string]string)
	if constantsText != "" {
		varNames := extractVariableNames(constantsText)
		rng := rand.New(rand.NewSource(seed))
		prefixes := []string{"in", "var", "tmp", "aux", "res", "out", "arg", "param", "local", "global"}
		// Also add single letters a-z.
		for ch := 'a'; ch <= 'z'; ch++ {
			prefixes = append(prefixes, string(ch))
		}
		newNames := make([]string, len(varNames))
		for i := range varNames {
			prefix := prefixes[rng.Intn(len(prefixes))]
			newNames[i] = fmt.Sprintf("%s_%d", prefix, i+1)
		}
		for i, v := range varNames {
			mapping[v] = newNames[i]
		}
	}

	var replaceFunc func(string) string
	if len(mapping) > 0 {
		var keys []string
		for k := range mapping {
			keys = append(keys, regexp.QuoteMeta(k))
		}
		pattern := regexp.MustCompile(`\b(` + strings.Join(keys, "|") + `)\b`)
		replaceFunc = func(text string) string {
			if text == "" {
				return text
			}
			return pattern.ReplaceAllStringFunc(text, func(match string) string {
				return mapping[match]
			})
		}
	} else {
		replaceFunc = func(text string) string {
			return text
		}
	}

	renamedTexts := make([]string, len(texts))
	for i, t := range texts {
		renamedTexts[i] = replaceFunc(t)
	}
	renamedConstants := ""
	if constantsText != "" {
		renamedConstants = replaceFunc(constantsText)
	}
	return renamedTexts, renamedConstants, mapping
}

// makePrefix constructs the prompt text given the examples and answer index.
func makePrefix(model, examples string, NQuestion int, instruct bool) string {
	if !strings.HasPrefix(model, "Qwen") {
		log.Fatal("Only Qwen models are supported.")
	}
	if instruct {
		return fmt.Sprintf(`<|im_start|>system
    You are a helpful assistant.
    You first think about the reasoning process in your mind and then provide the user with the answer.
    <|im_end|>
    <|im_start|>user
    Your role is to take a known pattern of symbolic constraints that represent the longest execution path of a program 
    and generalize it for any given input size N.
    When you receive an input value N,
    you must generate a canonical SMT-LIB constraint string that adheres to the following rules:
    (assert (op (op (op var_1 var_2)) (op (op var_3 var_4)) (op (op var_5 var_6)) (op var_7 var_8)))
    where op is a logical operator (e.g., 'and', 'or', 'not') and var_i are variables or constants.
    All per-variable constraints must be combined using a top-level (assert (and ...)) clause.
    The output must be in exact, canonical SMT-LIB format without extra commentary in the constraint string.
    Show your work in <think> </think> tags. And return the final SMT-LIB constraint string in <answer> </answer> tags.
    For example: <answer>(assert (and  ( >=  in0 97)  ( <=  in0 122)))</answer>.
    Here are the known constraints:
    %s
    What is the constraint for N=%d?
    <|im_end|>
    <|im_start|>assistant
    Let me solve this step by step.
    <think>`, examples, NQuestion)
	}
	return fmt.Sprintf(`A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Your role is to take a known pattern of symbolic constraints that represent the longest execution path of a program 
and generalize it for any given input size N.
When you receive an input value N,
you must generate a canonical SMT-LIB constraint string that adheres to the following rules:
(assert (op (op (op var_1 var_2)) (op (op var_3 var_4)) (op (op var_5 var_6)) (op var_7 var_8)))
where op is a logical operator (e.g., 'and', 'or', 'not') and var_i are variables or constants.
All per-variable constraints must be combined using a top-level (assert (and ...)) clause.
The output must be in exact, canonical SMT-LIB format without extra commentary in the constraint string.
Show your work in <think> </think> tags. And return the final SMT-LIB constraint string in <answer> </answer> tags.
For example: <answer>(assert (and  ( >=  in0 97)  ( <=  in0 122)))</answer>.
Here are the known constraints:
%s
What is the constraint for N=%d?
Assistant: Let me solve this step by step.
<think>`, examples, NQuestion)
}

// generateIndexPairs returns all (combo, answerIndex) pairs.
func generateIndexPairs(max_n, min_examples, max_examples, difference int) []IndexPair {
	var pairs []IndexPair
	candidateIndices := make([]int, max_n)
	for i := 0; i < max_n; i++ {
		candidateIndices[i] = i + 1
	}
	for k := min_examples; k <= max_examples; k++ {
		combs := combinations(candidateIndices, k)
		for _, combo := range combs {
			lastExample := combo[len(combo)-1]
			for d := 1; d <= difference; d++ {
				answerIndex := lastExample + d
				if answerIndex <= max_n {
					pairs = append(pairs, IndexPair{Combo: combo, AnswerIndex: answerIndex})
				}
			}
		}
	}
	return pairs
}

// combinations returns all combinations of length k from arr.
func combinations(arr []int, k int) [][]int {
	var res [][]int
	var comb func(start int, curr []int)
	comb = func(start int, curr []int) {
		if len(curr) == k {
			tmp := make([]int, k)
			copy(tmp, curr)
			res = append(res, tmp)
			return
		}
		for i := start; i < len(arr); i++ {
			comb(i+1, append(curr, arr[i]))
		}
	}
	comb(0, []int{})
	return res
}

// md5Seed computes an int64 seed from the MD5 hash of input.
func md5Seed(input string) int64 {
	h := md5.New()
	h.Write([]byte(input))
	sum := h.Sum(nil)
	var seed int64 = 0
	// Use the first 8 bytes.
	for i := 0; i < 8 && i < len(sum); i++ {
		seed = (seed << 8) | int64(sum[i])
	}
	return seed
}

// generateQuestionEntriesForProblem creates question entries for one problem.
func generateQuestionEntriesForProblem(files map[int]string, problem string, globalSeed int64, instruct bool, min_examples, max_examples, difference, max_n int) ([]QuestionEntry, error) {
	parsedFiles := make(map[int]SMT2File)
	for idx, path := range files {
		fileContent, err := parseSMT2File(path)
		if err != nil {
			return nil, err
		}
		parsedFiles[idx] = fileContent
	}
	var entries []QuestionEntry
	pairs := generateIndexPairs(max_n, min_examples, max_examples, difference)
	for _, pair := range pairs {
		comboStr := fmt.Sprintf("%v", pair.Combo)
		seedInput := fmt.Sprintf("%s_%s_%d_%d", problem, comboStr, pair.AnswerIndex, globalSeed)
		// Removed localSeed since it's unused:
		_ = md5Seed(seedInput)

		answerParsed, ok := parsedFiles[pair.AnswerIndex]
		if !ok {
			answerParsed = SMT2File{Constants: "", Solution: ""}
		}

		type Example struct {
			Index    int
			Solution string
		}
		var examples []Example
		for _, i := range pair.Combo {
			f, ok := parsedFiles[i]
			if !ok {
				f = SMT2File{Constants: "", Solution: ""}
			}
			examples = append(examples, Example{Index: i, Solution: f.Solution})
		}

		var allTexts []string
		for _, ex := range examples {
			allTexts = append(allTexts, ex.Solution)
		}
		allTexts = append(allTexts, answerParsed.Solution)

		renameSeedInput := fmt.Sprintf("%s_%s_%d_rename_%d", problem, comboStr, pair.AnswerIndex, globalSeed)
		renameSeed := md5Seed(renameSeedInput)

		renamedTexts, renamedAnswerConstants, mapping := renameVariablesInTexts(allTexts, answerParsed.Constants, renameSeed)

		// Build examples string and also record the indices.
		var renamedExamples []string
		var exampleIndices []string
		for i, ex := range examples {
			renamedExamples = append(renamedExamples, fmt.Sprintf("N=%d: %s", ex.Index, renamedTexts[i]))
			exampleIndices = append(exampleIndices, strconv.Itoa(ex.Index))
		}
		examplesStr := strings.Join(renamedExamples, "\n")
		questionText := makePrefix("Qwen", examplesStr, pair.AnswerIndex, instruct)

		// Prepare variable mapping as a simple comma-separated key:value string.
		var mappingParts []string
		for k, v := range mapping {
			mappingParts = append(mappingParts, fmt.Sprintf("%s:%s", k, v))
		}
		mappingStr := strings.Join(mappingParts, ",")

		entry := QuestionEntry{
			Problem:         problem,
			ExampleIndices:  strings.Join(exampleIndices, ","),
			Examples:        examplesStr,
			Question:        questionText,
			AnswerIndex:     pair.AnswerIndex,
			AnswerConstants: renamedAnswerConstants,
			AnswerSolution:  renamedTexts[len(renamedTexts)-1],
			VariableMapping: mappingStr,
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// collectSMT2DataFrame groups files by problem and generates all question entries.
func collectSMT2DataFrame(baseDir string, globalSeed int64, instruct bool) ([]QuestionEntry, error) {
	problemsFiles, err := groupSMT2Files(baseDir)
	if err != nil {
		return nil, err
	}
	var allEntries []QuestionEntry
	for problem, files := range problemsFiles {
		fmt.Printf("Processing problem: %s\n", problem)
		entries, err := generateQuestionEntriesForProblem(files, problem, globalSeed, instruct, 3, 5, 3, 30)
		if err != nil {
			return nil, err
		}
		allEntries = append(allEntries, entries...)
	}
	fmt.Printf("Collected %d question instances.\n", len(allEntries))
	return allEntries, nil
}

func main() {
	baseDir := flag.String("base_dir", "", "Base directory to search for SMT2 files")
	output := flag.String("output", "output.parquet", "Output Parquet file")
	seed := flag.Int64("seed", 69420, "Random seed for reproducibility")
	instruct := flag.Bool("instruct", false, "Whether to follow Qwen instruction format")
	flag.Parse()

	if *baseDir == "" {
		log.Fatal("base_dir is required")
	}

	entries, err := collectSMT2DataFrame(*baseDir, *seed, *instruct)
	if err != nil {
		log.Fatalf("Error collecting data: %v", err)
	}

	// Write the results to a Parquet file.
	fw, err := local.NewLocalFileWriter(*output)
	if err != nil {
		log.Fatalf("Can't create local file writer: %v", err)
	}
	pw, err := writer.NewParquetWriter(fw, new(QuestionEntry), 4)
	if err != nil {
		log.Fatalf("Can't create parquet writer: %v", err)
	}
	pw.RowGroupSize = 128 * 1024 * 1024
	pw.CompressionType = parquet.CompressionCodec_SNAPPY

	for _, entry := range entries {
		if err = pw.Write(entry); err != nil {
			log.Println("Write error", err)
		}
	}
	if err = pw.WriteStop(); err != nil {
		log.Fatalf("WriteStop error: %v", err)
	}
	fw.Close()
	fmt.Println("Parquet file written successfully.")
}
