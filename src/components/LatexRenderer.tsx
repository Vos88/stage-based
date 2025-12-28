import React from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

interface LatexRendererProps {
  content: string;
  className?: string;
}

export function LatexRenderer({ content, className = "" }: LatexRendererProps) {
  // Function to split content into text and math parts
  const parseContent = (text: string) => {
    const parts: Array<{ type: 'text' | 'inline-math' | 'block-math'; content: string }> = [];
    
    // Match block math: $$...$$
    const blockMathRegex = /\$\$([\s\S]*?)\$\$/g;
    let lastIndex = 0;
    let match;
    
    while ((match = blockMathRegex.exec(text)) !== null) {
      // Add text before the math
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.slice(lastIndex, match.index)
        });
      }
      
      // Add the block math
      parts.push({
        type: 'block-math',
        content: match[1].trim()
      });
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.slice(lastIndex)
      });
    }
    
    // Process each part for inline math
    const finalParts: Array<{ type: 'text' | 'inline-math' | 'block-math'; content: string }> = [];
    
    parts.forEach(part => {
      if (part.type === 'block-math') {
        finalParts.push(part);
      } else {
        // Process text for inline math: $...$
        const inlineMathRegex = /\$([^\$]+)\$/g;
        let lastIndex = 0;
        let match;
        
        while ((match = inlineMathRegex.exec(part.content)) !== null) {
          // Add text before the math
          if (match.index > lastIndex) {
            finalParts.push({
              type: 'text',
              content: part.content.slice(lastIndex, match.index)
            });
          }
          
          // Add the inline math
          finalParts.push({
            type: 'inline-math',
            content: match[1].trim()
          });
          
          lastIndex = match.index + match[0].length;
        }
        
        // Add remaining text
        if (lastIndex < part.content.length) {
          finalParts.push({
            type: 'text',
            content: part.content.slice(lastIndex)
          });
        }
      }
    });
    
    return finalParts;
  };

  const parts = parseContent(content);

  // Build lines preserving inline-math parts so we can detect lists
  type Part = { type: 'text' | 'inline-math' | 'block-math'; content: string };

  const lines: Part[][] = [];
  let currentLine: Part[] = [];

  parts.forEach(p => {
    if (p.type === 'text') {
      const segments = p.content.split(/\n/);
      segments.forEach((seg, i) => {
        if (seg.length > 0) currentLine.push({ type: 'text', content: seg });
        if (i < segments.length - 1) {
          // line break -> push current line
          lines.push(currentLine);
          currentLine = [];
        }
      });
    } else if (p.type === 'inline-math') {
      // inline math belongs to current line
      currentLine.push(p as Part);
    } else if (p.type === 'block-math') {
      // ensure any accumulated text is pushed as its own line
      if (currentLine.length > 0) {
        lines.push(currentLine);
        currentLine = [];
      }
      // push the block math as its own line
      lines.push([{ type: 'block-math', content: p.content }]);
    }
  });
  if (currentLine.length > 0) lines.push(currentLine);

  // Helper to render inline parts for a line
  const renderInlineParts = (partsForLine: Part[], keyBase: string) => (
    partsForLine.map((part, idx) => {
      const key = `${keyBase}-${idx}`;
      if (part.type === 'text') return <span key={key}>{part.content}</span>;
      if (part.type === 'inline-math') {
        try {
          return <InlineMath key={key} math={part.content} />;
        } catch (error) {
          console.error('Invalid inline math:', part.content, error);
          return <span key={key} className="text-red-500">${part.content}$</span>;
        }
      }
      // shouldn't reach here for block-math inside a line
      return null;
    })
  );

  // Render lines with simple list detection
  const rendered: React.ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // If this line is a standalone block math, render it in place
    if (line.length === 1 && line[0].type === 'block-math') {
      rendered.push(
        <div key={`bm-inline-${i}`} className="my-4 flex justify-center">
          <BlockMath math={line[0].content} />
        </div>
      );
      i++;
      continue;
    }
    // build string of the line's leading text to detect list markers
    const leadingText = line.find(p => p.type === 'text')?.content.trimStart() || '';

    const orderedMatch = leadingText.match(/^\d+\.\s+/);
    const unorderedMatch = leadingText.match(/^[-*]\s+/);

    if (orderedMatch) {
      // collect consecutive ordered lines
      const items: Part[][] = [];
      while (i < lines.length) {
        const l = lines[i];
        const firstText = l.find(p => p.type === 'text')?.content || '';
        if (!firstText.match(/^\d+\.\s+/)) break;
        // remove leading marker from the first text part
        const newParts: Part[] = [];
        let removed = false;
        l.forEach(p => {
          if (!removed && p.type === 'text') {
            const txt = p.content.replace(/^\d+\.\s+/, '');
            if (txt.length > 0) newParts.push({ type: 'text', content: txt });
            removed = true;
          } else {
            newParts.push(p);
          }
        });
        items.push(newParts);
        i++;
      }
      rendered.push(
        <ol key={`ol-${i}`} className="list-decimal pl-6">
          {items.map((partsItem, idx) => (
            <li key={idx} className="mb-1">
              {renderInlineParts(partsItem, `ol-${i}-${idx}`)}
            </li>
          ))}
        </ol>
      );
      continue;
    }

    if (unorderedMatch) {
      const items: Part[][] = [];
      while (i < lines.length) {
        const l = lines[i];
        const firstText = l.find(p => p.type === 'text')?.content || '';
        if (!firstText.match(/^[-*]\s+/)) break;
        const newParts: Part[] = [];
        let removed = false;
        l.forEach(p => {
          if (!removed && p.type === 'text') {
            const txt = p.content.replace(/^[-*]\s+/, '');
            if (txt.length > 0) newParts.push({ type: 'text', content: txt });
            removed = true;
          } else {
            newParts.push(p);
          }
        });
        items.push(newParts);
        i++;
      }
      rendered.push(
        <ul key={`ul-${i}`} className="list-disc pl-6">
          {items.map((partsItem, idx) => (
            <li key={idx} className="mb-1">
              {renderInlineParts(partsItem, `ul-${i}-${idx}`)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Regular paragraph/line: render inline parts and preserve single line breaks
    rendered.push(
      <p key={`p-${i}`} className="mb-3">
        {renderInlineParts(line, `p-${i}`)}
      </p>
    );
    i++;
  }

  // Final render
  return (
    <div className={className}>
      {rendered}
    </div>
  );
}
