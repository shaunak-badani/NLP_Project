import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import backendClient from "@/components/backendClient";

interface Chunk {
    chunk_number: number;
    text: string;
    relevance_score: number;
}

interface Response {
    answer: string;
    chunks: Chunk[];
}

const DeepLearning = () => {

    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState<Response | null>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [chunkingStrategy, setChunkingStrategy] = useState("sentence");
    const [embeddingModel, setEmbeddingModel] = useState("sentence-transformer");
    const [similarityMetric, setsimilarityMetric] = useState("cosine");
    const [uploadStatus, setUploadStatus] = useState("");
    const [tokenSize, setTokenSize] = useState(256);
    const [numChunks, setNumChunks] = useState(5);
    const [sentenceSize, setSentenceSize] = useState(1);
    const [paragraphSize, setParagraphSize] = useState(1);
    const [pageSize, setPageSize] = useState(1);
    const [visualizationImagePCA, setVisualizationImagePCA] = useState<string | null>(null);
    const [visualizationImageTSNE, setVisualizationImageTSNE] = useState<string | null>(null);
    const [visualizationImageUMAP, setVisualizationImageUMAP] = useState<string | null>(null);

    const [visualizationStatus, setVisualizationStatus] = useState("");


    const handlePromptInput = async(query: string) => {
        setLoading(true);
        try {
            const response = await backendClient.get("/deep-learning", {
                params: {
                    query: query,
                    num_chunks: numChunks
                }
            });
            setResponse(response.data);
        } catch (error) {
            setResponse({ answer: "Error processing your query. Please try again.", chunks: [] });
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
        } else if (chunkingStrategy === "sentence") {
            formData.append("sentence_size", sentenceSize.toString());
        } else if (chunkingStrategy === "paragraph") {
            formData.append("paragraph_size", paragraphSize.toString());
        } else if (chunkingStrategy === "page") {
            formData.append("page_size", pageSize.toString());
        }
        formData.append("embedding_model", embeddingModel);
        formData.append("similarity_metric", similarityMetric);

        try {
            const response = await backendClient.post("/upload-pdf", formData, {
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

    const handleVisualization = async () => {
        setLoading(true);
        try {
            // Call the API for PCA, t-SNE, and UMAP visualizations separately
            const pcaResponse = await backendClient.get("/visualize-embeddings", {
                params: { method: "pca", k: numChunks }, 
                responseType: 'json',
            });
    
            const tsneResponse = await backendClient.get("/visualize-embeddings", {
                params: { method: "tsne", k: numChunks }, 
                responseType: 'json',
            });
    
            const umapResponse = await backendClient.get("/visualize-embeddings", {
                params: { method: "umap", k: numChunks }, 
                responseType: 'json',
            });
    
            // Check if there are errors in any of the responses
            if (pcaResponse.data.error || tsneResponse.data.error || umapResponse.data.error) {
                setVisualizationStatus("Error generating visualizations.");
            } else {
                // Set state for each visualization
                setVisualizationImagePCA(pcaResponse.data.image);
                setVisualizationImageTSNE(tsneResponse.data.image);
                setVisualizationImageUMAP(umapResponse.data.image);
            }
        } catch (error) {
            if (error instanceof Error) {
                setVisualizationStatus(`Error generating visualization: ${error.message}`);
                console.error("Visualization error:", error);
            } else {
                setVisualizationStatus("Unknown error occurred while generating visualization");
            }
        }
        setLoading(false);
    };
    
    return (
        <>
            <div className="mb-8 p-6 border rounded-lg">
                <h3 className="text-lg font-medium mb-4">Document Upload</h3>
                <p className="mb-4">
                     NLP approach using deep learning, using LLM's
                </p>
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

                    {chunkingStrategy === "sentence" && (
                        <div>
                            <Label htmlFor="sentence-size" className="block mb-2">Number of Sentences per Chunk</Label>
                            <input
                                id="sentence-size"
                                type="number"
                                min="1"
                                max="10"
                                value={sentenceSize}
                                onChange={(e) => setSentenceSize(Number(e.target.value))}
                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                            />
                        </div>
                    )}

                    {chunkingStrategy === "paragraph" && (
                        <div>
                            <Label htmlFor="paragraph-size" className="block mb-2">Number of Paragraphs per Chunk</Label>
                            <input
                                id="paragraph-size"
                                type="number"
                                min="1"
                                max="5"
                                value={paragraphSize}
                                onChange={(e) => setParagraphSize(Number(e.target.value))}
                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                            />
                        </div>
                    )}

                    {chunkingStrategy === "page" && (
                        <div>
                            <Label htmlFor="page-size" className="block mb-2">Number of Pages per Chunk</Label>
                            <input
                                id="page-size"
                                type="number"
                                min="1"
                                max="3"
                                value={pageSize}
                                onChange={(e) => setPageSize(Number(e.target.value))}
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
                                <SelectItem value="fine-tuned-financial">Fine-tuned Financial (bge-base)</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    <div>
                        <Label htmlFor="similarity-metric" className="block mb-2">Similarity Metric</Label>
                        <Select 
                            value={similarityMetric} 
                            onValueChange={setsimilarityMetric}
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
                        <Label htmlFor="num-chunks" className="block mb-2">Number of Chunks to Retrieve</Label>
                        <input
                            id="num-chunks"
                            type="number"
                            min="1"
                            max="20"
                            value={numChunks}
                            onChange={(e) => setNumChunks(Number(e.target.value))}
                            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
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
            
            <div className="mb-8">
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
            
            {response && (
                <div className="space-y-6">
                    <div className="p-6 border rounded-lg bg-gray-50">
                        <h3 className="text-lg font-medium mb-2">Response</h3>
                        <div className="prose prose-sm md:prose-base lg:prose-lg dark:prose-invert prose-pre:bg-gray-800 prose-pre:text-gray-100 max-w-none">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {response.answer}
                            </ReactMarkdown>
                        </div>
                    </div>

                    {response.chunks && response.chunks.length > 0 && (
                        <div className="p-6 border rounded-lg">
                            <h3 className="text-lg font-medium mb-4">Relevant Chunks</h3>
                            <Accordion type="single" collapsible className="w-full">
                                {response.chunks.map((chunk) => (
                                    <AccordionItem key={chunk.chunk_number} value={`chunk-${chunk.chunk_number}`}>
                                        <AccordionTrigger>
                                            <div className="flex items-center justify-between w-full">
                                                <span>Chunk {chunk.chunk_number}</span>
                                                <span className="text-sm text-gray-500">
                                                    Relevance: {(chunk.relevance_score * 100).toFixed(2)}%
                                                </span>
                                            </div>
                                        </AccordionTrigger>
                                        <AccordionContent>
                                            <div className="prose prose-sm md:prose-base dark:prose-invert prose-pre:bg-gray-800 prose-pre:text-gray-100 max-w-none">
                                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                    {chunk.text}
                                                </ReactMarkdown>
                                            </div>
                                        </AccordionContent>
                                    </AccordionItem>
                                ))}
                            </Accordion>
                        </div>
                    )}
                </div>
            )}

        {response && (
            <div className="p-6 border rounded-lg bg-gray-50">
                <h3 className="text-lg font-medium mb-2">Visualizations</h3>
                
                <Button 
                    onClick={handleVisualization} 
                    className="mb-4"
                    disabled={isLoading}
                >
                    {isLoading ? "Generating..." : `Create visualization`}
                </Button>

                {visualizationStatus.toLowerCase().includes("error") && (
                    <div className="p-3 rounded-md bg-red-100 text-red-800">
                        {visualizationStatus}
                    </div>
                )}

                
                {visualizationImagePCA && (
                    <div id="visualization-container" className="mt-4 min-h-[200px] border rounded-lg p-4 bg-white">
                        <img 
                            src={visualizationImagePCA} 
                            alt="PCA Visualization" 
                            className="w-full border rounded-lg"
                        />
                    </div>
                )}

                {visualizationImageTSNE && (
                    <div id="visualization-container" className="mt-4 min-h-[200px] border rounded-lg p-4 bg-white">
                        <img 
                            src={visualizationImageTSNE} 
                            alt="t-SNE Visualization" 
                            className="w-full border rounded-lg"
                        />
                    </div>
                )}

                {visualizationImageUMAP && (
                    <div id="visualization-container" className="mt-4 min-h-[200px] border rounded-lg p-4 bg-white">
                        <img 
                            src={visualizationImageUMAP} 
                            alt="UMAP Visualization" 
                            className="w-full border rounded-lg"
                        />
                    </div>
                )}

            </div>
        )}


            
            
            {isLoading && <BackdropWithSpinner />}
        </>

        
    )
};

export default DeepLearning;