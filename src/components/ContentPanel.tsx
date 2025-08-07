import { ConceptNode } from "@/data/aiConceptTree";
import { Button } from "@/components/ui/button";
import { CheckCircle, AlertTriangle, ExternalLink } from "lucide-react";

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

      {node.content && (
        <div className="space-y-8">
          {/* Overview Section */}
          {node.content.overview && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">Overview</h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                {node.content.overview}
              </p>
            </section>
          )}

          {/* How It Works Section */}
          {node.content.howItWorks && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">How It Works</h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                {node.content.howItWorks}
              </p>
            </section>
          )}

          {/* Applications Section */}
          {node.content.applications && node.content.applications.length > 0 && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">Applications</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {node.content.applications.map((app, index) => (
                  <div key={index} className="p-4 rounded-lg bg-secondary/50 border border-border/50">
                    <p className="text-muted-foreground">{app}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Advantages Section */}
          {node.content.advantages && node.content.advantages.length > 0 && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">Advantages</h2>
              <ul className="space-y-3">
                {node.content.advantages.map((advantage, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <CheckCircle className="text-green-500 shrink-0 mt-0.5" size={20} />
                    <span className="text-muted-foreground">{advantage}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Limitations Section */}
          {node.content.limitations && node.content.limitations.length > 0 && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">Limitations</h2>
              <ul className="space-y-3">
                {node.content.limitations.map((limitation, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <AlertTriangle className="text-orange-500 shrink-0 mt-0.5" size={20} />
                    <span className="text-muted-foreground">{limitation}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Code Example Section */}
          {node.content.codeExample && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">Code Example</h2>
              <div className="code-block">
                <pre className="text-sm overflow-x-auto">
                  <code>{node.content.codeExample}</code>
                </pre>
              </div>
            </section>
          )}

          {/* Key Points Section (fallback for older content) */}
          {node.content.keyPoints && node.content.keyPoints.length > 0 && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">Key Points</h2>
              <ul className="space-y-2">
                {node.content.keyPoints.map((point, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span className="text-muted-foreground">{point}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Additional Resources Section */}
          {node.content.links && node.content.links.length > 0 && (
            <section>
              <h2 className="text-2xl font-semibold mb-4 text-foreground">Additional Resources</h2>
              <div className="space-y-3">
                {node.content.links.map((link, index) => (
                  <a
                    key={index}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 p-3 rounded-lg bg-secondary/50 border border-border/50 hover:bg-secondary/70 transition-colors text-primary hover:text-primary/80"
                  >
                    <span>{link.title}</span>
                    <ExternalLink size={16} />
                  </a>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}