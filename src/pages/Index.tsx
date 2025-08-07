import { ConceptExplorer } from "@/components/ConceptExplorer";
import { aiConceptTree } from "@/data/aiConceptTree";

const Index = () => {
  return (
    <ConceptExplorer rootNode={aiConceptTree} />
  );
};

export default Index;
