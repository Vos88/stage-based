import { ConceptNode } from "@/data/aiConceptTree";
import { Button } from "@/components/ui/button";
import { CheckCircle, AlertTriangle, ExternalLink } from "lucide-react";
import { LatexRenderer } from "./LatexRenderer";

interface ContentPanelProps {
  node: ConceptNode;
  onBack: () => void;
}

export function ContentPanel({ node, onBack }: ContentPanelProps) {
  return (
    <div className="content-card fade-in max-w-5xl w-full">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold text-foreground mb-3">
            {node.title}
          </h1>
          <p className="text-xl text-muted-foreground">
            {node.description}
          </p>
        </div>
        <Button variant="outline" onClick={onBack} className="shrink-0">
          ← Back
        </Button>
      </div>

      <div className="space-y-8">
        {/* Overview Section */}
        {node.overview && (
          <section>
            <h2 className="text-2xl font-semibold mb-4 text-foreground">Overview</h2>
            <div className="text-lg text-muted-foreground leading-relaxed">
              <LatexRenderer content={node.overview} />
            </div>
          </section>
        )}

        {/* How It Works Section */}
        {node.howItWorks && (
          <section>
            <h2 className="text-2xl font-semibold mb-4 text-foreground">How It Works</h2>
            <div className="text-lg text-muted-foreground leading-relaxed">
              <LatexRenderer content={node.howItWorks} />
            </div>
          </section>
        )}

        {/* Applications Section */}
        {node.applications && node.applications.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold mb-4 text-foreground">Applications</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {node.applications.map((app, index) => (
                <div key={index} className="p-4 rounded-lg bg-secondary/50 border border-border/50">
                  <div className="text-muted-foreground">
                    <LatexRenderer content={app} />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Advantages Section */}
        {node.advantages && node.advantages.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold mb-4 text-foreground">Advantages</h2>
            <ul className="space-y-3">
              {node.advantages.map((advantage, index) => (
                <li key={index} className="flex items-start gap-3">
                  <CheckCircle className="text-green-500 shrink-0 mt-0.5" size={20} />
                  <span className="text-muted-foreground">
                    <LatexRenderer content={advantage} />
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Limitations Section */}
        {node.limitations && node.limitations.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold mb-4 text-foreground">Limitations</h2>
            <ul className="space-y-3">
              {node.limitations.map((limitation, index) => (
                <li key={index} className="flex items-start gap-3">
                  <AlertTriangle className="text-orange-500 shrink-0 mt-0.5" size={20} />
                  <span className="text-muted-foreground">
                    <LatexRenderer content={limitation} />
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Code Example Section */}
        {node.codeExample && (
          <section>
            <h2 className="text-2xl font-semibold mb-4 text-foreground">Code Example</h2>
            <pre className="bg-slate-900 text-white p-4 rounded-md overflow-x-auto">
              <code>{node.codeExample}</code>
            </pre>
          </section>
        )}
      </div>
    </div>
  );
}