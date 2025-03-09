import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import axios from "axios";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";

const Mean = () => {
    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [chunkingStrategy, setChunkingStrategy] = useState("tokens");
    const [similarityMetric, setSimilarityMetric] = useState("cosine");
    const [uploadStatus, setUploadStatus] = useState("");
    const [tokenSize, setTokenSize] = useState(256);
    const [overlap, setOverlap] = useState(20);
    const [numResults, setNumResults] = useState(3);
    const [results, setResults] = useState<any[]>([]);
    const [serverUrl, setServerUrl] = useState("http://localhost:8080");

    const handlePromptInput = async(query: string) => {
        setLoading(true);
        try {
            const response = await axios.get(`${serverUrl}/mean-naive`, {
                params: {
                    query: query,
                    num_results: numResults,
                    similarity_metric: similarityMetric
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
        formData.append("chunking_strategy", chunkingStrategy);
        formData.append("token_size", tokenSize.toString());
        formData.append("overlap", overlap.toString());
        formData.append("similarity_metric", similarityMetric);

        try {
            const response = await axios.post(`${serverUrl}/upload-naive`, formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                }
            });
            
            if (response.data && response.data.message) {
                setUploadStatus(`Success! ${response.data.message}`);
            } else {
                setUploadStatus(`Success! Document processed with ${chunkingStrategy} chunking strategy.`);
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
                <h3 className="text-lg font-medium mb-4">Document Upload for Naive Approach</h3>
                <p className="mb-4">
                    This model uses TF-IDF vectorization with scikit-learn and basic text preprocessing 
                    for document retrieval. It's effective for keyword-based searches in smaller documents.
                </p>
                
                <div className="space-y-4">
                    <div>
                        <Label htmlFor="server-url" className="block mb-2">Server URL</Label>
                        <input
                            id="server-url"
                            type="text"
                            value={serverUrl}
                            onChange={(e) => setServerUrl(e.target.value)}
                            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                        />
                    </div>
                    
                    <div>
                        <Label htmlFor="pdf-upload" className="block mb-2">Upload PDF Document</Label>
                        <input 
                            id="pdf-upload"
                            type="file" 
                            accept=".pdf,.txt" 
                            onChange={handleFileChange}
                            className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                        />
                    </div>
                    
                    <div>
                        <Label htmlFor="chunking-strategy" className="block mb-2">Chunking Strategy</Label>
                        <Select 
                            value={chunkingStrategy} 
                            onValueChange={setChunkingStrategy}
                        >
                            <SelectTrigger className="w-full">
                                <SelectValue placeholder="Select chunking strategy" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="tokens">By Tokens</SelectItem>
                                <SelectItem value="sentence">By Sentence</SelectItem>
                                <SelectItem value="paragraph">By Paragraph</SelectItem>
                                <SelectItem value="page">By Page</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    
                    {chunkingStrategy === "tokens" && (
                        <>
                            <div>
                                <Label htmlFor="token-size" className="block mb-2">Token Size</Label>
                                <input
                                    id="token-size"
                                    type="number"
                                    min="50"
                                    max="1000"
                                    value={tokenSize}
                                    onChange={(e) => setTokenSize(Number(e.target.value))}
                                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                />
                            </div>
                            <div>
                                <Label htmlFor="overlap" className="block mb-2">Overlap</Label>
                                <input
                                    id="overlap"
                                    type="number"
                                    min="0"
                                    max="100"
                                    value={overlap}
                                    onChange={(e) => setOverlap(Number(e.target.value))}
                                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                />
                            </div>
                        </>
                    )}

                    <div>
                        <Label htmlFor="similarity-metric" className="block mb-2">Similarity Metric</Label>
                        <Select 
                            value={similarityMetric} 
                            onValueChange={setSimilarityMetric}
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
                        <Label htmlFor="num-results" className="block mb-2">Number of Results</Label>
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
                    placeholder="Enter your query here!"
                    className="mb-4"
                />
                <Button onClick={() => handlePromptInput(query)}>
                    Send Query
                </Button>
            </div>
            
            {response.length > 0 && (
                <div className="p-6 border rounded-lg bg-gray-50">
                    <h3 className="text-lg font-medium mb-2">Response</h3>
                    <p className="mb-4">{response}</p>
                    
                    {results.length > 0 && (
                        <div className="mt-4">
                            <h4 className="font-medium mb-2">Retrieved Chunks</h4>
                            <div className="space-y-3">
                                {results.map((result, index) => (
                                    <div key={index} className="p-3 border rounded bg-white">
                                        <div className="flex justify-between mb-1">
                                            <span className="font-medium">Chunk {result.chunk}/{result.total_chunks}</span>
                                            <span className="text-sm bg-blue-100 px-2 py-0.5 rounded">
                                                Score: {(result.score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <p className="text-sm">{result.snippet}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
            
            {isLoading && <BackdropWithSpinner />}
        </>
    )
};

export default Mean;