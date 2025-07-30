#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset of the vtk classes to be imported eagerly or lazily.
"""
from importlib import import_module

__all__ = []

######################################################################
location = {}
module_cache = {}

######################################################################
# noinspection PyUnresolvedReferences
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser

for name in [
    "vtkOpenGLGPUVolumeRayCastMapper",
    "vtkSmartVolumeMapper",
]: location[name] = "vtkRenderingVolumeOpenGL2"

######################################################################
for name in [
    "vtkKochanekSpline",
    "vtkCardinalSpline",
    "vtkParametricSpline",
    "vtkParametricFunctionSource",
    "vtkParametricTorus",
    "vtkParametricBoy",
    "vtkParametricConicSpiral",
    "vtkParametricCrossCap",
    "vtkParametricDini",
    "vtkParametricEllipsoid",
    "vtkParametricEnneper",
    "vtkParametricFigure8Klein",
    "vtkParametricKlein",
    "vtkParametricMobius",
    "vtkParametricRandomHills",
    "vtkParametricRoman",
    "vtkParametricSuperEllipsoid",
    "vtkParametricSuperToroid",
    "vtkParametricBohemianDome",
    "vtkParametricBour",
    "vtkParametricCatalanMinimal",
    "vtkParametricHenneberg",
    "vtkParametricKuen",
    "vtkParametricPluckerConoid",
    "vtkParametricPseudosphere",
]: location[name] = "vtkCommonComputationalGeometry"


location["vtkNamedColors"] = "vtkCommonColor"

location["vtkIntegrateAttributes"] = "vtkFiltersParallel"

# noinspection PyUnresolvedReferences
from vtkmodules.vtkCommonCore import (
    mutable,
    VTK_UNSIGNED_SHORT,
    VTK_UNSIGNED_INT,
    VTK_UNSIGNED_LONG,
    VTK_UNSIGNED_LONG_LONG,
    VTK_UNSIGNED_CHAR,
    VTK_CHAR,
    VTK_SHORT,
    VTK_INT,
    VTK_LONG,
    VTK_LONG_LONG,
    VTK_FLOAT,
    VTK_DOUBLE,
    VTK_SIGNED_CHAR,
    VTK_ID_TYPE,
    VTK_VERSION_NUMBER,
    VTK_FONT_FILE,
    vtkArray,
    vtkIdTypeArray,
    vtkBitArray,
    vtkCharArray,
    vtkCommand,
    vtkDoubleArray,
    vtkFloatArray,
    vtkIdList,
    vtkIntArray,
    vtkLookupTable,
    vtkPoints,
    vtkStringArray,
    vtkUnsignedCharArray,
    vtkVariant,
    vtkVariantArray,
    vtkVersion,
)
for name in [
    "mutable",
    "VTK_UNSIGNED_CHAR",
    "VTK_UNSIGNED_SHORT",
    "VTK_UNSIGNED_INT",
    "VTK_UNSIGNED_LONG",
    "VTK_UNSIGNED_LONG_LONG",
    "VTK_UNSIGNED_CHAR",
    "VTK_CHAR",
    "VTK_SHORT",
    "VTK_INT",
    "VTK_LONG",
    "VTK_LONG_LONG",
    "VTK_FLOAT",
    "VTK_DOUBLE",
    "VTK_SIGNED_CHAR",
    "VTK_ID_TYPE",
    "VTK_VERSION_NUMBER",
    "VTK_FONT_FILE",
    "vtkArray",
    "vtkIdTypeArray",
    "vtkBitArray",
    "vtkCharArray",
    "vtkCommand",
    "vtkDoubleArray",
    "vtkFloatArray",
    "vtkIdList",
    "vtkIntArray",
    "vtkLogger",
    "vtkLookupTable",
    "vtkMath",
    "vtkPoints",
    "vtkStringArray",
    "vtkUnsignedCharArray",
    "vtkVariant",
    "vtkVariantArray",
    "vtkVersion",
]: location[name] = "vtkCommonCore"

# noinspection PyUnresolvedReferences
from vtkmodules.vtkCommonDataModel import (
    vtkPolyData,
    vtkImageData,
    vtkUnstructuredGrid,
    vtkRectilinearGrid,
    vtkStructuredGrid,
    vtkCellArray,
    vtkDataSetAttributes,
    vtkDataObject,
    vtkDataSet,
    vtkFieldData,
    vtkHexagonalPrism,
    vtkHexahedron,
    vtkLine,
    vtkPentagonalPrism,
    vtkPixel,
    vtkPlane,
    vtkPlanes,
    vtkPointLocator,
    vtkPolyLine,
    vtkPolyPlane,
    vtkPolygon,
    vtkPolyVertex,
    vtkPyramid,
    vtkQuad,
    vtkTetra,
    vtkTriangle,
    vtkTriangleStrip,
    vtkVertex,
    vtkVoxel,
    vtkWedge,
)
for name in [
    "vtkCellArray",
    "vtkBox",
    "vtkCellLocator",
    "vtkCylinder",
    "vtkDataSetAttributes",
    "vtkDataObject",
    "vtkDataSet",
    "vtkFieldData",
    "vtkHexagonalPrism",
    "vtkHexahedron",
    "vtkImageData",
    "vtkImplicitDataSet",
    "vtkImplicitSelectionLoop",
    "vtkImplicitWindowFunction",
    # "vtkImplicitVolume",
    "vtkIterativeClosestPointTransform",
    "vtkLine",
    "vtkMultiBlockDataSet",
    "vtkMutableDirectedGraph",
    "vtkPentagonalPrism",
    "vtkPixel",
    "vtkPlane",
    "vtkPlaneCollection",
    "vtkPlanes",
    "vtkPointLocator",
    "vtkPolyData",
    "vtkPolyLine",
    "vtkPolyPlane",
    "vtkPolygon",
    "vtkPolyVertex",
    "vtkPyramid",
    "vtkQuad",
    "vtkQuadric",
    "vtkRectilinearGrid",
    "vtkSelection",
    "vtkSelectionNode",
    "vtkSphere",
    "vtkStaticCellLocator",
    "vtkStaticPointLocator",
    "vtkStructuredGrid",
    "vtkTetra",
    "vtkTriangle",
    "vtkTriangleStrip",
    "vtkUnstructuredGrid",
    "vtkVertex",
    "vtkVoxel",
    "vtkWedge",
]: location[name] = "vtkCommonDataModel"

# noinspection PyUnresolvedReferences
from vtkmodules.vtkCommonMath import vtkMatrix4x4
location["vtkAmoebaMinimizer"] = "vtkCommonMath"
location["vtkMatrix4x4"] = "vtkCommonMath"
location["vtkQuaternion"] = "vtkCommonMath"

# noinspection PyUnresolvedReferences
from vtkmodules.vtkCommonTransforms import (
    vtkHomogeneousTransform,
    vtkLandmarkTransform,
    vtkLinearTransform,
    vtkThinPlateSplineTransform,
    vtkTransform,
)
for name in [
    "vtkHomogeneousTransform",
    "vtkLandmarkTransform",
    "vtkLinearTransform",
    "vtkThinPlateSplineTransform",
    "vtkTransform",
]: location[name] = "vtkCommonTransforms"


for name in [
    "VTK_BEST_FITTING_PLANE",
    "vtk3DLinearGridCrinkleExtractor",
    "vtkAppendFilter",
    "vtkAppendPolyData",
    "vtkBinnedDecimation",
    "vtkCellCenters",
    "vtkCellDataToPointData",
    "vtkCenterOfMass",
    "vtkCleanPolyData",
    "vtkClipPolyData",
    "vtkPolyDataConnectivityFilter",
    "vtkPolyDataEdgeConnectivityFilter",
    "vtkContourFilter",
    "vtkContourGrid",
    "vtkCutter",
    "vtkDecimatePro",
    "vtkDelaunay2D",
    "vtkDelaunay3D",
    "vtkElevationFilter",
    "vtkFeatureEdges",
    "vtkFlyingEdges3D",
    "vtkGlyph3D",
    "vtkIdFilter",
    "vtkImageAppend",
    "vtkImplicitPolyDataDistance",
    "vtkMarchingSquares",
    "vtkMaskPoints",
    "vtkMassProperties",
    "vtkPointDataToCellData",
    "vtkPolyDataNormals",
    "vtkProbeFilter",
    "vtkQuadricClustering",
    "vtkQuadricDecimation",
    "vtkResampleWithDataSet",
    "vtkReverseSense",
    "vtkStripper",
    "vtkSurfaceNets3D",
    "vtkTensorGlyph",
    "vtkThreshold",
    "vtkTriangleFilter",
    "vtkTubeFilter",
    "vtkUnstructuredGridQuadricDecimation",
    "vtkVoronoi2D",
    "vtkWindowedSincPolyDataFilter",
    "vtkStaticCleanUnstructuredGrid",
    "vtkPolyDataPlaneCutter"
]: location[name] = "vtkFiltersCore"

# noinspection PyUnresolvedReferences
from vtkmodules.vtkFiltersCore import vtkGlyph3D


for name in [
    "vtkExtractCellsByType",
    "vtkExtractGeometry",
    "vtkExtractPolyDataGeometry",
    "vtkExtractSelection",
]: location[name] = "vtkFiltersExtraction"


location["vtkExtractEdges"] = "vtkFiltersCore"

location["vtkStreamTracer"] = "vtkFiltersFlowPaths"


for name in [
    "vtkBooleanOperationPolyDataFilter",
    "vtkBoxClipDataSet",
    "vtkCellValidator",
    "vtkClipDataSet",
    "vtkClipClosedSurface",
    "vtkCountVertices",
    "vtkContourTriangulator",
    "vtkCurvatures",
    "vtkDataSetTriangleFilter",
    "vtkDensifyPolyData",
    "vtkDistancePolyDataFilter",
    "vtkGradientFilter",
    "vtkIntersectionPolyDataFilter",
    "vtkLoopBooleanPolyDataFilter",
    "vtkMultiBlockDataGroupFilter",
    "vtkPolyDataToReebGraphFilter",
    "vtkTransformPolyDataFilter",
    "vtkTransformFilter",
    "vtkOBBTree",
    "vtkQuantizePolyDataPoints",
    "vtkRandomAttributeGenerator",
    "vtkShrinkFilter",
    "vtkShrinkPolyData",
    "vtkRectilinearGridToTetrahedra",
    "vtkVertexGlyphFilter",
]: location[name] = "vtkFiltersGeneral"

try:
    from vtkmodules.vtkCommonDataModel import vtkCellTreeLocator
    location["vtkCellTreeLocator"] = "vtkCommonDataModel"
except ImportError:
    from vtkmodules.vtkFiltersGeneral import vtkCellTreeLocator
    location["vtkCellTreeLocator"] = "vtkFiltersGeneral"


for name in [
    "vtkAttributeSmoothingFilter",
    "vtkDataSetSurfaceFilter",
    "vtkGeometryFilter",
    "vtkImageDataGeometryFilter",
    "vtkMarkBoundaryFilter",
]: location[name] = "vtkFiltersGeometry"


for name in [
    "vtkFacetReader",
    "vtkImplicitModeller",
    "vtkPolyDataSilhouette",
    "vtkProcrustesAlignmentFilter",
    "vtkRenderLargeImage",
]: location[name] = "vtkFiltersHybrid"


for name in [
    "vtkAdaptiveSubdivisionFilter",
    "vtkBandedPolyDataContourFilter",
    "vtkButterflySubdivisionFilter",
    "vtkContourLoopExtraction",
    "vtkCookieCutter",
    "vtkDijkstraGraphGeodesicPath",
    "vtkFillHolesFilter",
    "vtkHausdorffDistancePointSetFilter",
    "vtkImprintFilter",
    "vtkLinearExtrusionFilter",
    "vtkLinearSubdivisionFilter",
    "vtkLoopSubdivisionFilter",
    "vtkRibbonFilter",
    "vtkRotationalExtrusionFilter",
    "vtkRuledSurfaceFilter",
    "vtkSectorSource",
    "vtkSelectEnclosedPoints",
    "vtkSelectPolyData",
    "vtkSubdivideTetra",
    "vtkTrimmedExtrusionFilter",
]: location[name] = "vtkFiltersModeling"


for name in [
    "vtkConnectedPointsFilter",
    "vtkDensifyPointCloudFilter",
    "vtkEuclideanClusterExtraction",
    "vtkExtractEnclosedPoints",
    "vtkExtractSurface",
    "vtkGaussianKernel",
    "vtkLinearKernel",
    "vtkPCANormalEstimation",
    "vtkPointDensityFilter",
    "vtkPointInterpolator",
    "vtkRadiusOutlierRemoval",
    "vtkShepardKernel",
    "vtkSignedDistance",
    "vtkPointSmoothingFilter",
    "vtkUnsignedDistance",
    "vtkVoronoiKernel",
]: location[name] = "vtkFiltersPoints"


for name in [
    "vtkArcSource",
    "vtkArrowSource",
    "vtkConeSource",
    "vtkCubeSource",
    "vtkCylinderSource",
    "vtkDiskSource",
    "vtkFrustumSource",
    "vtkGlyphSource2D",
    "vtkGraphToPolyData",
    "vtkLineSource",
    "vtkOutlineCornerFilter",
    "vtkParametricFunctionSource",
    "vtkPlaneSource",
    "vtkPointSource",
    "vtkProgrammableSource",
    "vtkSphereSource",
    "vtkTexturedSphereSource",
    "vtkTessellatedBoxSource",
]: location[name] = "vtkFiltersSources"

location["vtkTextureMapToPlane"] = "vtkFiltersTexture"

location["vtkMeshQuality"] = "vtkFiltersVerdict"
location["vtkCellSizeFilter"] = "vtkFiltersVerdict"

location["vtkPolyDataToImageStencil"] = "vtkImagingStencil"

location["vtkX3DExporter"] = "vtkIOExport"

location["vtkGL2PSExporter"] = "vtkIOExportGL2PS"

for name in [
    "vtkBYUReader",
    "vtkFacetWriter",
    "vtkOBJReader",
    "vtkOpenFOAMReader",
    "vtkParticleReader",
    "vtkSTLReader",
    "vtkSTLWriter",
]: location[name] = "vtkIOGeometry"

for name in [
    "vtkBMPReader",
    "vtkBMPWriter",
    "vtkDEMReader",
    "vtkDICOMImageReader",
    "vtkHDRReader",
    "vtkJPEGReader",
    "vtkJPEGWriter",
    "vtkMetaImageReader",
    "vtkMetaImageWriter",
    "vtkNIFTIImageReader",
    "vtkNIFTIImageWriter",
    "vtkNrrdReader",
    "vtkPNGReader",
    "vtkPNGWriter",
    "vtkSLCReader",
    "vtkTIFFReader",
    "vtkTIFFWriter",
]: location[name] = "vtkIOImage"

location["vtk3DSImporter"] = "vtkIOImport"
location["vtkOBJImporter"] = "vtkIOImport"
location["vtkVRMLImporter"] = "vtkIOImport"

for name in [
    "vtkSimplePointsWriter",
    "vtkStructuredGridReader",
    "vtkStructuredPointsReader",
    "vtkDataSetReader",
    "vtkDataSetWriter",
    "vtkPolyDataWriter",
    "vtkRectilinearGridReader",
    "vtkUnstructuredGridReader",
]: location[name] = "vtkIOLegacy"


location["vtkPLYReader"] = "vtkIOPLY"
location["vtkPLYWriter"] = "vtkIOPLY"

for name in [
    "vtkXMLGenericDataObjectReader",
    "vtkXMLImageDataReader",
    "vtkXMLImageDataWriter",
    "vtkXMLMultiBlockDataReader",
    "vtkXMLMultiBlockDataWriter",
    "vtkXMLPRectilinearGridReader",
    "vtkXMLPUnstructuredGridReader",
    "vtkXMLPolyDataReader",
    "vtkXMLPolyDataWriter",
    "vtkXMLRectilinearGridReader",
    "vtkXMLRectilinearGridWriter",
    "vtkXMLStructuredGridReader",
    "vtkXMLUnstructuredGridReader",
    "vtkXMLUnstructuredGridWriter",
]: location[name] = "vtkIOXML"


location["vtkImageLuminance"] = "vtkImagingColor"
location["vtkImageMapToWindowLevelColors"] = "vtkImagingColor"

for name in [
    "vtkImageAppendComponents",
    "vtkImageBlend",
    "vtkImageCast",
    "vtkImageConstantPad",
    "vtkImageExtractComponents",
    "vtkImageFlip",
    "vtkImageMapToColors",
    "vtkImageMirrorPad",
    "vtkImagePermute",
    "vtkImageResample",
    "vtkImageResize",
    "vtkImageReslice",
    "vtkImageThreshold",
    "vtkImageTranslateExtent",
    "vtkExtractVOI",
]: location[name] = "vtkImagingCore"


for name in [
    "vtkImageButterworthHighPass",
    "vtkImageButterworthLowPass",
    "vtkImageFFT",
    "vtkImageFourierCenter",
    "vtkImageRFFT",
]: location[name] = "vtkImagingFourier"


for name in [
    "vtkImageCorrelation",
    "vtkImageEuclideanDistance",
    "vtkImageGaussianSmooth",
    "vtkImageGradient",
    "vtkImageHybridMedian2D",
    "vtkImageLaplacian",
    "vtkImageMedian3D",
    "vtkImageNormalize",
    "vtkImageSlab",
]: location[name] = "vtkImagingGeneral"

for name in ["vtkImageToPoints", "vtkSampleFunction"]:
    location[name] = "vtkImagingHybrid"


for name in [
    "vtkImageDivergence",
    "vtkImageDotProduct",
    "vtkImageLogarithmicScale",
    "vtkImageLogic",
    "vtkImageMagnitude",
    "vtkImageMathematics",
]: location[name] = "vtkImagingMath"

for name in [
    "vtkImageContinuousDilate3D",
    "vtkImageContinuousErode3D",
]: location[name] = "vtkImagingMorphological"

location["vtkImageCanvasSource2D"] = "vtkImagingSources"

location["vtkImageStencil"] = "vtkImagingStencil"

for name in [
    "vtkCircularLayoutStrategy",
    "vtkClustering2DLayoutStrategy",
    "vtkConeLayoutStrategy",
    "vtkFast2DLayoutStrategy",
    "vtkForceDirectedLayoutStrategy",
    "vtkGraphLayout",
    "vtkSimple2DLayoutStrategy",
    "vtkSimple3DCirclesStrategy",
    "vtkSpanTreeLayoutStrategy",
]: location[name] = "vtkInfovisLayout"

for name in [
    "vtkInteractorStyleFlight",
    "vtkInteractorStyleImage",
    "vtkInteractorStyleJoystickActor",
    "vtkInteractorStyleJoystickCamera",
    "vtkInteractorStyleRubberBand2D",
    "vtkInteractorStyleRubberBand3D",
    "vtkInteractorStyleRubberBandZoom",
    "vtkInteractorStyleTerrain",
    "vtkInteractorStyleTrackballActor",
    "vtkInteractorStyleTrackballCamera",
    "vtkInteractorStyleUnicam",
    "vtkInteractorStyleUser",
]: location[name] = "vtkInteractionStyle"

# noinspection PyUnresolvedReferences
from vtkmodules.vtkInteractionWidgets import (
    vtkBalloonWidget,
    vtkBoxWidget,
    vtkContourWidget,
    vtkFocalPlanePointPlacer,
    vtkImplicitPlaneWidget,
    vtkOrientationMarkerWidget,
    vtkOrientedGlyphContourRepresentation,
    vtkPlaneWidget,
    vtkPolygonalSurfacePointPlacer,
    vtkSliderWidget,
    vtkSphereWidget,
)
for name in [
    "vtkBalloonRepresentation",
    "vtkBalloonWidget",
    "vtkBoxWidget",
    "vtkButtonWidget",
    "vtkContourWidget",
    "vtkPlaneWidget",
    "vtkFocalPlanePointPlacer",
    "vtkImageTracerWidget",
    "vtkImplicitPlaneWidget",
    "vtkOrientationMarkerWidget",
    "vtkOrientedGlyphContourRepresentation",
    "vtkPolygonalSurfacePointPlacer",
    "vtkSliderRepresentation2D",
    "vtkSliderRepresentation3D",
    "vtkSliderWidget",
    "vtkSphereWidget",
    "vtkTexturedButtonRepresentation2D",
]: location[name] = "vtkInteractionWidgets"

location["vtkCameraOrientationWidget"] = "vtkInteractionWidgets"

# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingAnnotation import (
    vtkAxesActor,
    vtkAxisActor2D,
    vtkCaptionActor2D,
    vtkCornerAnnotation,
    vtkLegendBoxActor,
    vtkLegendScaleActor,
    vtkScalarBarActor,
)
for name in [
    "vtkAnnotatedCubeActor",
    "vtkArcPlotter",
    "vtkAxesActor",
    "vtkAxisActor2D",
    "vtkCaptionActor2D",
    "vtkCornerAnnotation",
    "vtkCubeAxesActor",
    "vtkLegendBoxActor",
    "vtkLegendScaleActor",
    "vtkPolarAxesActor",
    "vtkScalarBarActor",
    "vtkXYPlotActor",
]: location[name] = "vtkRenderingAnnotation"


# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkAreaPicker,
    vtkAssembly,
    vtkBillboardTextActor3D,
    vtkCamera,
    vtkCoordinate,
    vtkDataSetMapper,
    vtkFlagpoleLabel,
    vtkFollower,
    vtkImageActor,
    vtkImageProperty,
    vtkImageSlice,
    vtkInteractorObserver,
    vtkLight,
    vtkLogLookupTable,
    vtkProp,
    vtkPropAssembly,
    vtkPropCollection,
    vtkPropPicker,
    vtkProperty,
    vtkRenderWindow,
    vtkRenderer,
    vtkRenderWindowInteractor,
    vtkTextActor,
    vtkTextProperty,
    vtkTexture,
    vtkViewport,
    vtkVolume,
    vtkVolumeProperty,
)
for name in [
    "vtkActor",
    "vtkActor2D",
    "vtkAreaPicker",
    "vtkAssembly",
    "vtkBillboardTextActor3D",
    "vtkCamera",
    "vtkCameraInterpolator",
    "vtkColorTransferFunction",
    "vtkCoordinate",
    "vtkDataSetMapper",
    "vtkDistanceToCamera",
    "vtkFlagpoleLabel",
    "vtkFollower",
    "vtkHierarchicalPolyDataMapper",
    "vtkImageActor",
    "vtkImageMapper",
    "vtkImageProperty",
    "vtkImageSlice",
    "vtkInteractorEventRecorder",
    "vtkInteractorObserver",
    "vtkLight",
    "vtkLightKit",
    "vtkLogLookupTable",
    "vtkMapper",
    "vtkPointGaussianMapper",
    "vtkPolyDataMapper",
    "vtkPolyDataMapper2D",
    "vtkProp",
    "vtkProp3D",
    "vtkPropAssembly",
    "vtkPropCollection",
    "vtkPropPicker",
    "vtkProperty",
    "vtkRenderWindow",
    "vtkRenderer",
    "vtkRenderWindowInteractor",
    "vtkSelectVisiblePoints",
    "vtkSkybox",
    "vtkTextActor",
    "vtkTextMapper",
    "vtkTextProperty",
    "vtkTextRenderer",
    "vtkTexture",
    "vtkTransformInterpolator",
    "vtkViewport",
    "vtkVolume",
    "vtkVolumeProperty",
    "vtkWindowToImageFilter",
]: location[name] = "vtkRenderingCore"

location["vtkVectorText"] = "vtkRenderingFreeType"

location["vtkImageResliceMapper"] = "vtkRenderingImage"

location["vtkLabeledDataMapper"] = "vtkRenderingLabel"

for name in [
    "vtkDepthOfFieldPass",
    "vtkCameraPass",
    "vtkDualDepthPeelingPass",
    "vtkEquirectangularToCubeMapTexture",
    "vtkLightsPass",
    "vtkOpaquePass",
    "vtkOverlayPass",
    "vtkRenderPassCollection",
    "vtkSSAOPass",
    "vtkSequencePass",
    "vtkShader",
    "vtkShadowMapPass",
    "vtkTranslucentPass",
    "vtkVolumetricPass",
]: location[name] = "vtkRenderingOpenGL2"


for name in [
    "vtkFixedPointVolumeRayCastMapper",
    "vtkGPUVolumeRayCastMapper",
]: location[name] = "vtkRenderingVolume"

###########################################################################
# https://vtk.org/doc/nightly/html/vtkCellType_8h.html
cell_types = {
    "EMPTY_CELL": 0,
    "VERTEX": 1,
    "POLY_VERTEX": 2,
    "LINE": 3,
    "POLY_LINE": 4,
    "TRIANGLE": 5,
    "TRIANGLE_STRIP": 6,
    "POLYGON": 7,
    "PIXEL": 8,
    "QUAD": 9,
    "TETRA": 10,
    "VOXEL": 11,
    "HEXAHEDRON": 12,
    "WEDGE": 13,
    "PYRAMID": 14,
    "PENTAGONAL_PRISM": 15,
    "HEXAGONAL_PRISM": 16,
    "QUADRATIC_EDGE": 21,
    "QUADRATIC_TRIANGLE": 22,
    "QUADRATIC_QUAD": 23,
    "QUADRATIC_POLYGON": 36,
    "QUADRATIC_TETRA": 24,
    "QUADRATIC_HEXAHEDRON": 25,
    "QUADRATIC_WEDGE": 26,
    "QUADRATIC_PYRAMID": 27,
    "BIQUADRATIC_QUAD": 28,
    "TRIQUADRATIC_HEXAHEDRON": 29,
    "TRIQUADRATIC_PYRAMID": 37,
    "QUADRATIC_LINEAR_QUAD": 30,
    "QUADRATIC_LINEAR_WEDGE": 31,
    "BIQUADRATIC_QUADRATIC_WEDGE": 32,
    "BIQUADRATIC_QUADRATIC_HEXAHEDRON": 33,
    "BIQUADRATIC_TRIANGLE": 34,
    "CUBIC_LINE": 35,
    "CONVEX_POINT_SET": 41,
    "POLYHEDRON": 42,
    "PARAMETRIC_CURVE": 51,
    "PARAMETRIC_SURFACE": 52,
    "PARAMETRIC_TRI_SURFACE": 53,
    "PARAMETRIC_QUAD_SURFACE": 54,
    "PARAMETRIC_TETRA_REGION": 55,
    "PARAMETRIC_HEX_REGION": 56,
    "HIGHER_ORDER_EDGE": 60,
    "HIGHER_ORDER_TRIANGLE": 61,
    "HIGHER_ORDER_QUAD": 62,
    "HIGHER_ORDER_POLYGON": 63,
    "HIGHER_ORDER_TETRAHEDRON": 64,
    "HIGHER_ORDER_WEDGE": 65,
    "HIGHER_ORDER_PYRAMID": 66,
    "HIGHER_ORDER_HEXAHEDRON": 67,
    "LAGRANGE_CURVE": 68,
    "LAGRANGE_TRIANGLE": 69,
    "LAGRANGE_QUADRILATERAL": 70,
    "LAGRANGE_TETRAHEDRON": 71,
    "LAGRANGE_HEXAHEDRON": 72,
    "LAGRANGE_WEDGE": 73,
    "LAGRANGE_PYRAMID": 74,
    "BEZIER_CURVE": 75,
    "BEZIER_TRIANGLE": 76,
    "BEZIER_QUADRILATERAL": 77,
    "BEZIER_TETRAHEDRON": 78,
    "BEZIER_HEXAHEDRON": 79,
    "BEZIER_WEDGE": 80,
    "BEZIER_PYRAMID": 81,
}

###########################################################################
array_types = {}
array_types[VTK_UNSIGNED_CHAR] = "uint8"
array_types[VTK_UNSIGNED_SHORT]= "uint16"
array_types[VTK_UNSIGNED_INT]  = "uint32"
array_types[VTK_UNSIGNED_LONG_LONG] = "uint64"
array_types[VTK_CHAR]          = "int8"
array_types[VTK_SHORT]         = "int16"
array_types[VTK_INT]           = "int32"
# array_types[VTK_LONG]  # ??
array_types[VTK_LONG_LONG]     = "int64"
array_types[VTK_FLOAT]         = "float32"
array_types[VTK_DOUBLE]        = "float64"
array_types[VTK_SIGNED_CHAR]   = "int8"
array_types[VTK_ID_TYPE]       = "int64"
############ reverse aliases
array_types["char"]            = VTK_UNSIGNED_CHAR
array_types["uint8"]           = VTK_UNSIGNED_CHAR
array_types["uint16"]          = VTK_UNSIGNED_SHORT
array_types["uint32"]          = VTK_UNSIGNED_INT
array_types["uint64"]          = VTK_UNSIGNED_LONG_LONG
array_types["int8"]            = VTK_CHAR
array_types["int16"]           = VTK_SHORT
array_types["int32"]           = VTK_INT
array_types["int64"]           = VTK_LONG_LONG
array_types["float32"]         = VTK_FLOAT
array_types["float64"]         = VTK_DOUBLE
array_types["int"]             = VTK_INT
array_types["float"]           = VTK_FLOAT
############ reverse aliases
array_types["UNSIGNED_CHAR"]  = VTK_UNSIGNED_CHAR
array_types["UNSIGNED_SHORT"] = VTK_UNSIGNED_SHORT
array_types["UNSIGNED_INT"]   = VTK_UNSIGNED_INT
array_types["UNSIGNED_LONG_LONG"] = VTK_UNSIGNED_LONG_LONG
array_types["CHAR"]           = VTK_CHAR
array_types["SHORT"]          = VTK_SHORT
array_types["INT"]            = VTK_INT
array_types["LONG"]           = VTK_LONG
array_types["LONG_LONG"]      = VTK_LONG_LONG
array_types["FLOAT"]          = VTK_FLOAT
array_types["DOUBLE"]         = VTK_DOUBLE
array_types["SIGNED_CHAR"]    = VTK_SIGNED_CHAR
array_types["ID_TYPE"]        = VTK_ID_TYPE


#########################################################
from vedo.settings import Settings

if Settings.dry_run_mode < 2:
    # https://vtk.org/doc/nightly/html
    # /md__builds_gitlab_kitware_sciviz_ci_Documentation_Doxygen_PythonWrappers.html
    # noinspection PyUnresolvedReferences
    import vtkmodules.vtkRenderingOpenGL2
    # noinspection PyUnresolvedReferences
    import vtkmodules.vtkInteractionStyle
    # noinspection PyUnresolvedReferences
    import vtkmodules.vtkRenderingFreeType
    # noinspection PyUnresolvedReferences
    import vtkmodules.vtkRenderingVolumeOpenGL2


#########################################################
def cell_type_names():
    """
    Return a dict of cell type names.
    Eg. `cell_type_names[10]` returns "TETRA".
    """
    # invert the dict above to get a lookup table for cell types
    return {v: k for k, v in cell_types.items()}


######################################################################
def get_class(name, module_name=""):
    """
    Get a vtk class from its name.

    Example:
    ```python
    from vedo import vtkclasses as vtki
    print(vtkActor)
    print(location["vtkActor"])
    print(get_class("vtkActor"))
    print(get_class("vtkActor", "vtkRenderingCore"))
    ```
    """
    if name and not name.lower().startswith("vtk"):
        name = "vtk" + name
    if not module_name:
        module_name = location[name]
    module_name = "vtkmodules." + module_name
    if module_name not in module_cache:
        module = import_module(module_name)
        module_cache[module_name] = module
    if name:
        return getattr(module_cache[module_name], name)
    else:
        return module_cache[module_name]

######################################################################
def new(cls_name, module_name=""):
    """
    Create a new vtk object instance from its name.

    Example:
    ```python
    from vedo import vtkclasses as vtki
    a = new("Actor")
    ```
    """
    try:
        instance = get_class(cls_name, module_name)()
    except NotImplementedError as e:
        print(e, cls_name)
        return None
    return instance

######################################################################
def dump_hierarchy_to_file(fname=""):
    """
    Print all available vtk classes.
    Dumps the list to a file named `vtkmodules_<version>_hierarchy.txt`
    in the current working directory.

    Example:
    ```python
    from vedo.vtkclasses import dump_hierarchy_to_file
    dump_hierarchy_to_file()
    ```
    """
    try:
        import pkgutil
        import vtkmodules
        from vtkmodules.all import vtkVersion
        ver = vtkVersion()
    except AttributeError:
        print("Unable to detect VTK version.")
        return
    major = ver.GetVTKMajorVersion()
    minor = ver.GetVTKMinorVersion()
    patch = ver.GetVTKBuildVersion()
    vtkvers = f"{major}.{minor}.{patch}"
    if not fname:
        fname = f"vtkmodules_{vtkvers}_hierarchy.txt"
    with open(fname,"w") as w:
        for pkg in pkgutil.walk_packages(
            vtkmodules.__path__, vtkmodules.__name__ + "."):
            try:
                module = import_module(pkg.name)
            except ImportError:
                continue
            for subitem in sorted(dir(module)):
                if "all" in module.__name__:
                    continue
                if ".web." in module.__name__:
                    continue
                if ".test." in module.__name__:
                    continue
                if ".tk." in module.__name__:
                    continue
                if "__" in module.__name__ or "__" in subitem:
                    continue
                w.write(f"{module.__name__}.{subitem}\n")

#########################################################
# print("successfully finished importing vtkmodules")
#########################################################
