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

  // Render lines with hierarchical nested list detection
  type Token = { indent: number; marker: 'ol' | 'ul' | null; parts: Part[] };

  const tokens: Token[] = lines.map(l => {
    const firstText = l.find(p => p.type === 'text')?.content || '';
    const match = firstText.match(/^(\s*)(\d+\.\s+|[-*]\s+)?(.*)$/);
    const indent = match ? match[1].length : 0;
    const markerRaw = match ? match[2] : null;
    const marker = markerRaw ? (markerRaw.trim().endsWith('.') ? 'ol' : 'ul') : null;

    // Build new parts array where the first text part has the leading marker removed
    const newParts: Part[] = [];
    let removed = false;
    l.forEach(p => {
      if (!removed && p.type === 'text') {
        // Use the part after the marker (the third capture group) if present
        const contentAfter = match ? match[3] : p.content;
        if (contentAfter.length > 0) newParts.push({ type: 'text', content: contentAfter });
        removed = true;
      } else {
        newParts.push(p);
      }
    });

    return { indent, marker, parts: newParts };
  });

  // Convert tokens into a tree of nodes: paragraphs, nested lists, and block-math
  type ListItem = { parts: Part[]; children: Array<any> };
  type ListNode = { nodeType: 'list'; listType: 'ol' | 'ul'; level: number; items: ListItem[] };
  type ParaNode = { nodeType: 'para'; parts: Part[] };
  type BlockMathNode = { nodeType: 'block-math'; content: string };
  const nodes: Array<ListNode | ParaNode | BlockMathNode> = [];

  let currentList: ListNode | null = null;

  const findListAtLevel = (level: number, listType: 'ol' | 'ul') => {
    // Search nodes from the end for a list with matching level and type
    for (let j = nodes.length - 1; j >= 0; j--) {
      const n = nodes[j] as any;
      if (n.nodeType === 'list' && n.level === level && n.listType === listType) return n as ListNode;
    }
    return null;
  };

  tokens.forEach(tok => {
    const level = Math.floor(tok.indent / 2);
    // If this token represents a standalone block math, render as block-math node
    if (!tok.marker && tok.parts.length === 1 && tok.parts[0].type === 'block-math') {
      currentList = null;
      nodes.push({ nodeType: 'block-math', content: tok.parts[0].content });
      return;
    }

    if (!tok.marker) {
      // plain paragraph line
      currentList = null;
      nodes.push({ nodeType: 'para', parts: tok.parts });
      return;
    }

    if (!currentList) {
      // start a new top-level list
      const newList: ListNode = { nodeType: 'list', listType: tok.marker, level, items: [{ parts: tok.parts, children: [] }] };
      nodes.push(newList);
      currentList = newList;
      return;
    }

    if (level === currentList.level && tok.marker === currentList.listType) {
      // same list level and type
      currentList.items.push({ parts: tok.parts, children: [] });
      return;
    }

    if (level > currentList.level) {
      // nested list: attach to last item of currentList
      const lastItem = currentList.items[currentList.items.length - 1];
      if (!lastItem) {
        // ensure there's at least one item
        currentList.items.push({ parts: [], children: [] });
      }
      const nestedList: ListNode = { nodeType: 'list', listType: tok.marker, level, items: [{ parts: tok.parts, children: [] }] };
      // attach nested list under the lastItem.children
      (currentList.items[currentList.items.length - 1].children ||= []).push(nestedList);
      currentList = nestedList;
      return;
    }

    // level < currentList.level or different list type: try to find an existing list at that level
    const found = findListAtLevel(level, tok.marker);
    if (found) {
      found.items.push({ parts: tok.parts, children: [] });
      currentList = found;
      return;
    }

    // fallback: create a new top-level list
    const fallback: ListNode = { nodeType: 'list', listType: tok.marker, level, items: [{ parts: tok.parts, children: [] }] };
    nodes.push(fallback);
    currentList = fallback;
  });

  // Recursive render for nodes
  const renderParts = renderInlineParts;

  const renderListNode = (list: ListNode, keyBase: string) => {
    const Tag = list.listType === 'ol' ? 'ol' : 'ul';
    return (
      <Tag key={keyBase} className={list.listType === 'ol' ? 'list-decimal pl-6' : 'list-disc pl-6'}>
        {list.items.map((it, idx) => (
          <li key={`${keyBase}-li-${idx}`} className="mb-1">
            {renderParts(it.parts, `${keyBase}-item-${idx}`)}
            {it.children && it.children.length > 0 && it.children.map((child: any, cidx: number) => (
              // child is a ListNode
              renderListNode(child as ListNode, `${keyBase}-child-${idx}-${cidx}`)
            ))}
          </li>
        ))}
      </Tag>
    );
  };

  const rendered: React.ReactNode[] = nodes.map((n, idx) => {
    if ((n as ParaNode).nodeType === 'para') {
      return (
        <p key={`p-${idx}`} className="mb-3">
          {renderParts((n as ParaNode).parts, `p-${idx}`)}
        </p>
      );
    }
    if ((n as BlockMathNode).nodeType === 'block-math') {
      return (
        <div key={`bm-${idx}`} className="my-4 flex justify-center">
          <BlockMath math={(n as BlockMathNode).content} />
        </div>
      );
    }
    return renderListNode(n as ListNode, `list-${idx}`);
  });

  // Final render
  return (
    <div className={className}>
      {rendered}
    </div>
  );
}
