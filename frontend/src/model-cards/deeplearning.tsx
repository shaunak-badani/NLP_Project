import { useState } from "react";
import { Button } from "@/components/ui/button";
import axios from "axios";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";

const DeepLearning = () => {

    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [chunkingStrategy, setChunkingStrategy] = useState("sentence");
    const [embeddingModel, setEmbeddingModel] = useState("sentence-transformer");
    const [uploadStatus, setUploadStatus] = useState("");
    const [tokenSize, setTokenSize] = useState(256);

    const handlePromptInput = async(query: string) => {
        setLoading(true);
        try {
            const response = await axios.get("http://localhost:8000/api/deep-learning", {
                params: {
                    query: query
                }
            });
            setResponse(response.data.answer);
        } catch (error) {
            setResponse("Error processing your query. Please try again.");
            console.error("Error:", error);
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
        
        if (chunkingStrategy === "tokens") {
            formData.append("token_size", tokenSize.toString());
        }
        formData.append("embedding_model", embeddingModel);

        try {
            const response = await axios.post("http://localhost:8000/api/upload-pdf", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                }
            });
            setUploadStatus(`Success! ${response.data.message}`);
        } catch (error) {
            setUploadStatus("Error uploading file. Please try again.");
            console.error("Error:", error);
        }
        setLoading(false);
    }

    return (
        <>
            <div className="mb-8 p-6 border rounded-lg">
                <h3 className="text-lg font-medium mb-4">Document Upload</h3>
                
                <div className="space-y-4">
                    <div>
                        <Label htmlFor="pdf-upload" className="block mb-2">Upload PDF Document</Label>
                        <input 
                            id="pdf-upload"
                            type="file" 
                            accept=".pdf" 
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
                                <SelectItem value="sentence">By Sentence</SelectItem>
                                <SelectItem value="paragraph">By Paragraph</SelectItem>
                                <SelectItem value="page">By Page</SelectItem>
                                <SelectItem value="tokens">By Tokens</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    
                    {chunkingStrategy === "tokens" && (
                        <div>
                            <Label htmlFor="token-size" className="block mb-2">Token Size</Label>
                            <input
                                id="token-size"
                                type="number"
                                min="50"
                                max="1000"
                                value={tokenSize}
                                onChange={(e) => setTokenSize(Number(e.target.value))}
                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                            />
                        </div>
                    )}

                    <div>
                        <Label htmlFor="embedding-model" className="block mb-2">Embedding Model</Label>
                        <Select 
                            value={embeddingModel} 
                            onValueChange={setEmbeddingModel}
                        >
                            <SelectTrigger className="w-full">
                                <SelectValue placeholder="Select embedding model" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="sentence-transformer">Sentence-Transformer (all-MiniLM-L6-v2)</SelectItem>
                                <SelectItem value="bert">BERT (bert-base-uncased)</SelectItem>
                                <SelectItem value="roberta">RoBERTa (roberta-base)</SelectItem>
                                <SelectItem value="distilbert">DistilBERT (distilbert-base-uncased)</SelectItem>
                                <SelectItem value="gpt2">GPT-2 (gpt2)</SelectItem>
                            </SelectContent>
                        </Select>
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
                    <p>{response}</p>
                </div>
            )}
            
            {isLoading && <BackdropWithSpinner />}
        </>
    )
};

export default DeepLearning;