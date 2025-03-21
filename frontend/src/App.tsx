import './App.css'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card } from "@/components/ui/card"
import DeepLearning from './model-cards/deeplearning'
import Traditional from './model-cards/traditional'
import Mean from './model-cards/mean'

function App() {

  return (
    <>
    <div>
      <div className="header p-6 text-xl border-b">RAG Analyzer</div>
    <div className="min-h-screen p-8 pb-8 sm:p-8">      
      <main className="max-w-4xl mx-auto flex flex-col gap-16">
      <div>
      <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
      Making RAG Retrieval Transparent & Explainable
      </h1>
      <p className="leading-7 [&:not(:first-child)]:mt-6 m-6 sm:m-6">
      How to decide whether your RAG pipeline is truly optimized? RAG analyzer helps you visually analyze chunking strategies and embedding models to understand how well your system retrieves relevant information. Compare different methods and gain insights into which configuration works best for your data!
      </p>
      <Tabs defaultValue="mean">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="mean">Naive model</TabsTrigger>
          <TabsTrigger value="traditional">Traditional Model</TabsTrigger>
          <TabsTrigger value="deep-learning">Deep Learning Model</TabsTrigger>
        </TabsList>
        <TabsContent value="mean">
          <Card className="p-20">
            <Mean />
          </Card>
        </TabsContent>
        <TabsContent value="traditional">
          <Card className="p-20">
            <Traditional />
          </Card>
        </TabsContent>
        <TabsContent value="deep-learning">
          <Card className="p-20">
            <DeepLearning />
          </Card>
        </TabsContent>
      </Tabs>
      </div>


      </main>

    </div>
    </div>

      
    </>
  )
}

export default App
