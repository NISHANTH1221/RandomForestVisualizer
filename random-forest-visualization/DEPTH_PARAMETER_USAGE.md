# Depth Parameter Usage Guide

This document explains how to use the new depth parameter feature added to the `/api/trees/{id}` endpoint.

## Overview

The depth parameter allows you to fetch tree structures up to a specified depth, which is useful for:
- Performance optimization when dealing with large trees
- UI visualization that only needs to show a limited number of levels
- Progressive loading of tree data
- Memory management in frontend applications

## API Usage

### Backend API

```http
GET /api/trees/{tree_id}?depth={max_depth}
```

**Parameters:**
- `tree_id` (required): ID of the tree to extract (0 to n_estimators-1)
- `depth` (optional): Maximum depth to fetch (1-20, omit for full tree)

**Examples:**
```http
GET /api/trees/0           # Full tree
GET /api/trees/0?depth=3   # Limited to depth 3
GET /api/trees/5?depth=5   # Tree 5, limited to depth 5
```

### Frontend API Service

```typescript
import { getTreeStructure } from '@/services/api';

// Get full tree
const fullTree = await getTreeStructure(0);

// Get tree limited to depth 3
const limitedTree = await getTreeStructure(0, 3);

// Get tree limited to depth 5
const shallowTree = await getTreeStructure(0, 5);
```

## Response Format

The API response includes additional metadata when using depth limitation:

```json
{
  "success": true,
  "tree_id": 0,
  "depth_limit": 3,
  "extraction_info": {
    "timestamp": "2025-06-20T01:55:00Z",
    "format_version": "1.1",
    "api_version": "1.1.0",
    "depth_limited": true,
    "effective_max_depth": 3,
    "truncated_nodes": 7
  },
  "tree_data": {
    "tree_id": 0,
    "format_version": "1.1",
    "metadata": {
      "total_nodes": 15,
      "max_depth": 54,
      "effective_max_depth": 3,
      "is_depth_limited": true,
      "depth_limit": 3
    },
    "statistics": {
      "leaf_count": 8,
      "internal_node_count": 7,
      "truncated_nodes": 7,
      "original_total_nodes": 2733,
      "filtered_total_nodes": 15
    },
    "flat_nodes": [...],
    "root_node": {...}
  }
}
```

## Key Features

### 1. Node Truncation
- Nodes at the maximum depth are marked as truncated if they originally had children
- `is_truncated`: Boolean indicating if the node was truncated
- `truncated_children_count`: Number of children that were cut off

### 2. Performance Benefits
Based on test results with a tree containing 2,733 nodes:
- Full tree: ~0.050s extraction time
- Depth 6: ~0.014s extraction time
- Depth 4: ~0.012s extraction time
- Depth 2: ~0.012s extraction time

### 3. Memory Efficiency
- Depth 1: 3/2733 nodes (99.9% reduction)
- Depth 2: 7/2733 nodes (99.7% reduction)
- Depth 3: 15/2733 nodes (99.5% reduction)
- Depth 4: 29/2733 nodes (98.9% reduction)
- Depth 5: 47/2733 nodes (98.3% reduction)

## Validation Rules

### Backend Validation
- `depth` must be between 1 and 20 (inclusive)
- `depth` = 0 or negative values are rejected with HTTP 400
- `depth` > 20 is rejected to prevent performance issues
- Invalid `tree_id` still returns HTTP 400

### Frontend Validation
- The frontend API service accepts `depth` as an optional parameter
- `undefined` or `null` depth values result in full tree extraction
- Type safety ensures only numbers are passed as depth values

## Error Handling

### Invalid Depth Values
```http
GET /api/trees/0?depth=0
# Returns: HTTP 400 - "Invalid depth parameter: 0. Must be >= 1"

GET /api/trees/0?depth=25
# Returns: HTTP 400 - "Invalid depth parameter: 25. Must be <= 20"
```

### Invalid Tree ID
```http
GET /api/trees/999?depth=3
# Returns: HTTP 400 - "Invalid tree ID: 999. Must be between 0 and {n_estimators-1}"
```

## Use Cases

### 1. Tree Visualization Components
```typescript
// Progressive loading for large trees
const [currentDepth, setCurrentDepth] = useState(3);
const [treeData, setTreeData] = useState(null);

const loadTreeData = async (depth: number) => {
  const data = await getTreeStructure(treeId, depth);
  setTreeData(data);
};

// Load more levels on demand
const expandTree = () => {
  setCurrentDepth(prev => prev + 2);
  loadTreeData(currentDepth + 2);
};
```

### 2. Performance Optimization
```typescript
// Start with shallow tree for quick initial render
useEffect(() => {
  // Load shallow tree first
  getTreeStructure(treeId, 3).then(setInitialTree);
  
  // Load full tree in background
  setTimeout(() => {
    getTreeStructure(treeId).then(setFullTree);
  }, 100);
}, [treeId]);
```

### 3. Mobile-Friendly Views
```typescript
// Adjust depth based on screen size
const isMobile = useMediaQuery('(max-width: 768px)');
const maxDepth = isMobile ? 3 : 6;

const treeData = await getTreeStructure(treeId, maxDepth);
```

## Implementation Details

### Backend Changes
1. **TreeStructureExtractor**: Added `convert_tree_to_json_with_depth_limit()` method
2. **API Endpoint**: Enhanced `/api/trees/{tree_id}` to accept optional `depth` parameter
3. **Validation**: Added depth parameter validation (1-20 range)
4. **Metadata**: Enhanced response with depth-related metadata

### Frontend Changes
1. **API Service**: Updated `getTreeStructure()` to accept optional depth parameter
2. **Type Safety**: Maintained TypeScript compatibility
3. **URL Construction**: Proper query parameter handling

## Testing

The implementation includes comprehensive tests covering:
- ✅ Depth constraint verification
- ✅ Node truncation accuracy
- ✅ Hierarchical structure integrity
- ✅ Performance comparisons
- ✅ Edge case handling
- ✅ API parameter validation

Run tests with:
```bash
cd random-forest-visualization/backend
source venv/bin/activate
python test_depth_parameter.py
```

## Migration Guide

### Existing Code Compatibility
The depth parameter is optional, so existing code continues to work:

```typescript
// This still works (returns full tree)
const tree = await getTreeStructure(treeId);

// This is the new functionality
const limitedTree = await getTreeStructure(treeId, 3);
```

### Recommended Migration
For better performance, consider adding depth limits to existing tree visualizations:

```typescript
// Before
const tree = await getTreeStructure(treeId);

// After (recommended for large trees)
const tree = await getTreeStructure(treeId, 5);
```

## Best Practices

1. **Start Small**: Begin with depth 3-5 for initial renders
2. **Progressive Loading**: Allow users to expand deeper levels on demand
3. **Performance Monitoring**: Monitor extraction times and adjust depth limits accordingly
4. **User Experience**: Provide visual indicators when trees are truncated
5. **Error Handling**: Always handle depth validation errors gracefully

## Future Enhancements

Potential future improvements:
- Depth-based caching strategies
- Streaming/pagination for very large trees
- Dynamic depth adjustment based on tree complexity
- Client-side depth filtering for already-loaded trees
