import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import backendClient from "@/components/backendClient";

const Traditional = () => {
    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [chunkingMethod, setChunkingMethod] = useState("paragraph");
    const [similarityMethod, setSimilarityMethod] = useState("cosine");
    const [uploadStatus, setUploadStatus] = useState("");
    const [chunkSize, setChunkSize] = useState(200);
    const [overlap, setOverlap] = useState(50);
    const [sentencesPerChunk, setSentencesPerChunk] = useState(1);
    const [numResults, setNumResults] = useState(3);
    const [results, setResults] = useState<any[]>([]);

    const handleQuery = async(query: string) => {
        setLoading(true);
        try {
            const response = await backendClient.get(`/search-naive`, {
                params: {
                    query: query,
                    num_results: numResults,
                    similarity_method: similarityMethod
                }
            });
            
            if (response.data.answer) {
                setResponse(response.data.answer);
            }
            
            if (response.data.results) {
                setResults(response.data.results);
            } else {
                setResults([]);
            }
        } catch (error) {
            console.error("Query error:", error);
            setResponse("Error processing your query. Please check console for details.");
            setResults([]);
        }
        setLoading(false);
    }

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
            setUploadStatus("");
        }
    }

    const handleFileUpload = async () => {
        if (!selectedFile) {
            setUploadStatus("Please select a file first.");
            return;
        }

        setLoading(true);
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("chunking_method", chunkingMethod);
        formData.append("chunk_size", chunkSize.toString());
        formData.append("overlap", overlap.toString());
        formData.append("sentences_per_chunk", sentencesPerChunk.toString());

        try {
            const response = await backendClient.post(`/upload-naive`, formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                }
            });
            
            if (response.data && response.data.message) {
                setUploadStatus(`Success! ${response.data.message}`);
            } else {
                setUploadStatus(`Success! Document processed.`);
            }
        } catch (error) {
            console.error("Upload error:", error);
            setUploadStatus("Error uploading file. Please check console for details.");
        }
        setLoading(false);
    };

    return (
        <>
            <div className="mb-8 p-6 border rounded-lg">
                <h3 className="text-lg font-medium mb-4">Document Upload</h3>
                <p className="mb-4">
                    It applies TF-IDF vectorization with stemming and text processing
                </p>
                
                <div className="space-y-4">
                    <div>
                        <Label htmlFor="file-upload" className="block mb-2">Upload PDF Document</Label>
                        <input 
                            id="file-upload"
                            type="file" 
                            accept=".pdf,.txt" 
                            onChange={handleFileChange}
                            className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                        />
                    </div>
                    
                    <div>
                        <Label htmlFor="chunking-method" className="block mb-2">Chunking Strategy</Label>
                        <Select 
                            value={chunkingMethod} 
                            onValueChange={setChunkingMethod}
                        >
                            <SelectTrigger className="w-full">
                                <SelectValue placeholder="Select chunking strategy" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="paragraph">By Paragraph</SelectItem>
                                <SelectItem value="sentence">By Sentence</SelectItem>
                                <SelectItem value="fixed_size">By Fixed Size</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    
                    {chunkingMethod === "fixed_size" && (
                        <>
                            <div>
                                <Label htmlFor="chunk-size" className="block mb-2">Chunk Size (words)</Label>
                                <input
                                    id="chunk-size"
                                    type="number"
                                    min="50"
                                    max="1000"
                                    value={chunkSize}
                                    onChange={(e) => setChunkSize(Number(e.target.value))}
                                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                />
                            </div>
                            <div>
                                <Label htmlFor="overlap" className="block mb-2">Overlap (words)</Label>
                                <input
                                    id="overlap"
                                    type="number"
                                    min="0"
                                    max="200"
                                    value={overlap}
                                    onChange={(e) => setOverlap(Number(e.target.value))}
                                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                />
                            </div>
                        </>
                    )}

                    {chunkingMethod === "sentence" && (
                        <div>
                            <Label htmlFor="sentences-per-chunk" className="block mb-2">Number of Sentences per Chunk</Label>
                            <input
                                id="sentences-per-chunk"
                                type="number"
                                min="1"
                                max="10"
                                value={sentencesPerChunk}
                                onChange={(e) => setSentencesPerChunk(Number(e.target.value))}
                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                            />
                        </div>
                    )}

                    <div>
                        <Label htmlFor="similarity-method" className="block mb-2">Similarity Metric</Label>
                        <Select 
                            value={similarityMethod} 
                            onValueChange={setSimilarityMethod}
                        >
                            <SelectTrigger className="w-full">
                                <SelectValue placeholder="Select similarity metric" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="cosine">Cosine Similarity</SelectItem>
                                <SelectItem value="euclidean">Euclidean Similarity</SelectItem>
                                <SelectItem value="jaccard">Jaccard Similarity</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    <div>
                        <Label htmlFor="num-results" className="block mb-2">Number of Chunks to Retrieve</Label>
                        <input
                            id="num-results"
                            type="number"
                            min="1"
                            max="10"
                            value={numResults}
                            onChange={(e) => setNumResults(Number(e.target.value))}
                            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                        />
                    </div>
                    
                    <Button onClick={handleFileUpload} className="w-full">
                        Upload and Process
                    </Button>
                    
                    {uploadStatus && (
                        <div className={`p-3 rounded-md ${uploadStatus.includes("Success") ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                            {uploadStatus}
                        </div>
                    )}
                </div>
            </div>
            
            <div className="mb-8 p-6 border rounded-lg">
                <h3 className="text-lg font-medium mb-4">Query Document</h3>
                <Textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your query here..."
                    className="mb-4"
                />
                <Button onClick={() => handleQuery(query)}>
                    Search
                </Button>
            </div>
            
            {response.length > 0 && (
                <div className="p-6 border rounded-lg bg-gray-50">
                    <h3 className="text-lg font-medium mb-2">Response</h3>
                    <p className="mb-4">{response}</p>
                    
                    {results.length > 0 && (
                        <div className="mt-4">
                            <h4 className="font-medium mb-2">Retrieved Chunks</h4>
                            <Accordion type="single" collapsible className="w-full">
                                {results.map((result, index) => (
                                    <AccordionItem key={index} value={`chunk-${index}`}>
                                        <AccordionTrigger>
                                            <div className="flex items-center justify-between w-full">
                                                <span>Chunk {result.chunk}/{result.total_chunks}</span>
                                                <span className="text-sm bg-blue-100 px-2 py-0.5 rounded">
                                                    Score: {(result.score * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </AccordionTrigger>
                                        <AccordionContent>
                                            <p className="whitespace-pre-line">{result.snippet}</p>
                                        </AccordionContent>
                                    </AccordionItem>
                                ))}
                            </Accordion>
                        </div>
                    )}
                </div>
            )}
            
            {isLoading && <BackdropWithSpinner />}
        </>
    );
};

export default Traditional;