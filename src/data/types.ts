export interface ConceptNode {
  id: string;
  title: string;
  description: string;
  color: string;
  children?: ConceptNode[];
  overview?: string;
  howItWorks?: string;
  applications?: string[];
  advantages?: string[];
  limitations?: string[];
  codeExample?: string;
}

