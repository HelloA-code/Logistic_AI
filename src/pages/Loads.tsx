import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Package, Truck, Clock, CheckCircle, Search, Filter, Plus, ArrowRight, Sparkles, Brain, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import { IMAGES } from '@/assets/images';
import { Layout } from '@/components/Layout';
import { MetricCard, LoadCard } from '@/components/Cards';
import { LoadAssignmentForm, PostLoadForm } from '@/components/Forms';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { Load, LoadStatus } from '@/lib/index';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription 
} from '@/components/ui/dialog';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';

const getStatusColor = (status: LoadStatus) => {
  switch (status) {
    case 'pending': return 'bg-amber-500/10 text-amber-500 border-amber-500/20';
    case 'assigned': return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
    case 'in_transit': return 'bg-primary-500/10 text-primary border-primary/20';
    case 'delivered': return 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20';
    default: return 'bg-muted text-muted-foreground';
  }
};

export default function Loads() {
  const { loads, vehicles, assignLoad, addLoad, evaluateLoad, isLoading } = useRealtimeData();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLoadId, setSelectedLoadId] = useState<string | null>(null);
  const [postLoadOpen, setPostLoadOpen] = useState(false);
  const [evalResult, setEvalResult] = useState<any>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalLoadId, setEvalLoadId] = useState<string | null>(null);

  const handleEvaluateLoad = async (load: Load) => {
    setEvalLoading(true);
    setEvalLoadId(load.id);
    try {
      // Pick the best available vehicle (idle or loading, with capacity)
      const availableVehicle = vehicles.find(v => 
        (v.status === 'idle' || v.status === 'loading') && 
        (v.capacity_kg - v.current_load_kg) >= load.weight_kg
      ) || vehicles[0];
      
      const result = await evaluateLoad(load, availableVehicle);
      setEvalResult(result);
    } catch (err) {
      setEvalResult(null);
    } finally {
      setEvalLoading(false);
    }
  };

  const pendingLoads = loads.filter(l => l.status === 'pending');
  const activeLoads = loads.filter(l => l.status === 'assigned' || l.status === 'in_transit');
  const totalRevenue = loads.reduce((sum, l) => sum + l.price_inr, 0);
  const averagePrice = loads.length > 0 ? totalRevenue / loads.length : 0;

  const filteredLoads = loads.filter(l => 
    l.origin.toLowerCase().includes(searchQuery.toLowerCase()) || 
    l.destination.toLowerCase().includes(searchQuery.toLowerCase()) ||
    l.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (isLoading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-[calc(100vh-200px)]">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-8 pb-12">
        <section className="relative h-64 rounded-2xl overflow-hidden">
          <img 
            src={IMAGES.WAREHOUSE_OPS_3} 
            alt="Warehouse Operations" 
            className="absolute inset-0 w-full h-full object-cover opacity-40"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/40 to-transparent" />
          <div className="absolute bottom-8 left-8">
            <h1 className="text-4xl font-bold tracking-tight mb-2">Load Management</h1>
            <p className="text-muted-foreground max-w-xl">
              Real-time dispatching and load tracking across the NH-44 and NH-48 corridors. 
              Optimize your empty miles with AI-assisted matching.
            </p>
          </div>
        </section>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard 
            title="Available Loads"
            value={pendingLoads.length.toString()}
            icon={<Package className="h-5 w-5 text-amber-500" />}
            change="+12% from yesterday"
          />
          <MetricCard 
            title="In Transit"
            value={activeLoads.length.toString()}
            icon={<Truck className="h-5 w-5 text-primary" />}
            change="On schedule"
          />
          <MetricCard 
            title="Avg. Load Value"
            value={`₹${Math.round(averagePrice).toLocaleString()}`}
            icon={<Clock className="h-5 w-5 text-emerald-500" />}
          />
          <MetricCard 
            title="Total Pipeline"
            value={`₹${totalRevenue.toLocaleString()}`}
            icon={<CheckCircle className="h-5 w-5 text-blue-500" />}
            change="Active contracts"
          />
        </div>

        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-semibold flex items-center gap-2">
              <Package className="h-6 w-6 text-primary" />
              Pending Assignments
            </h2>
            <Button variant="outline" size="sm">
              View Market Map
            </Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {pendingLoads.length > 0 ? (
              pendingLoads.map((load) => (
                <div key={load.id} className="group relative">
                  <LoadCard load={load} />
                  {/* Show AI result badge if evaluated */}
                  {evalResult && evalLoadId === load.id && !evalLoading && (
                    <div className={`absolute top-2 right-2 z-10 px-2 py-1 rounded-md text-xs font-bold ${
                      evalResult.decision === 'ACCEPT' 
                        ? 'bg-emerald-500/90 text-white' 
                        : 'bg-red-500/90 text-white'
                    }`}>
                      {evalResult.decision === 'ACCEPT' ? '✓ ACCEPT' : '✗ SKIP'}
                    </div>
                  )}
                  <div className="absolute inset-0 bg-background/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 rounded-xl backdrop-blur-sm">
                    <Button 
                      onClick={() => setSelectedLoadId(load.id)}
                      className="shadow-lg"
                    >
                      Assign Vehicle
                    </Button>
                    <Button 
                      variant="secondary"
                      className="shadow-lg gap-1"
                      disabled={evalLoading && evalLoadId === load.id}
                      onClick={() => handleEvaluateLoad(load)}
                    >
                      {evalLoading && evalLoadId === load.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Brain className="h-4 w-4" />
                      )}
                      AI Evaluate
                    </Button>
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-full py-12 text-center border-2 border-dashed border-border rounded-xl">
                <p className="text-muted-foreground">No pending loads found. High fleet utilization detected.</p>
              </div>
            )}
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl overflow-hidden shadow-sm">
          <div className="p-6 border-b border-border flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold">All Loads</h2>
              <p className="text-sm text-muted-foreground">Complete history of shipments and pending offers</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input 
                  placeholder="Search routes..."
                  className="pl-10 w-[240px] bg-background/50"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <Button variant="outline" size="icon">
                <Filter className="h-4 w-4" />
              </Button>
              <Button size="sm" className="gap-2" onClick={() => setPostLoadOpen(true)}>
                <Plus className="h-4 w-4" /> Post Load
              </Button>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-[100px]">ID</TableHead>
                  <TableHead>Route</TableHead>
                  <TableHead>Weight</TableHead>
                  <TableHead>Price</TableHead>
                  <TableHead>Deadline</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredLoads.map((load) => (
                  <TableRow key={load.id} className="group">
                    <TableCell className="font-mono text-xs">{load.id}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{load.origin}</span>
                        <ArrowRight className="h-3 w-3 text-muted-foreground" />
                        <span className="font-medium">{load.destination}</span>
                      </div>
                    </TableCell>
                    <TableCell>{(load.weight_kg / 1000).toFixed(1)} Tons</TableCell>
                    <TableCell className="font-mono">₹{load.price_inr.toLocaleString()}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{load.delivery_deadline}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className={getStatusColor(load.status)}>
                        {load.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button variant="ghost" size="sm">
                        Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </div>

      <Dialog open={!!selectedLoadId} onOpenChange={(open) => !open && setSelectedLoadId(null)}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Assign Vehicle to Load</DialogTitle>
            <DialogDescription>
              Select an available vehicle from your fleet to fulfill this shipment. AI suggests vehicles based on proximity and fuel efficiency.
            </DialogDescription>
          </DialogHeader>
          {selectedLoadId && (
            <LoadAssignmentForm 
              loadId={selectedLoadId} 
              onAssign={(vehicleId) => {
                assignLoad(selectedLoadId, vehicleId);
                setSelectedLoadId(null);
              }} 
            />
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={postLoadOpen} onOpenChange={setPostLoadOpen}>
        <DialogContent className="sm:max-w-[550px]">
          <DialogHeader>
            <DialogTitle>Post New Load</DialogTitle>
            <DialogDescription>
              Create a new shipment order. It will appear as "Pending" and can be assigned to a vehicle.
            </DialogDescription>
          </DialogHeader>
          <PostLoadForm 
            onSubmit={(data) => {
              addLoad({
                origin: data.origin,
                destination: data.destination,
                weight_kg: data.weight_kg,
                price_inr: data.price_inr,
                pickup_date: data.pickup_date,
                delivery_deadline: data.delivery_deadline,
              });
              setPostLoadOpen(false);
            }} 
          />
        </DialogContent>
      </Dialog>

      {/* AI Load Evaluation Result Dialog */}
      <Dialog open={!!evalResult && !evalLoading} onOpenChange={(open) => !open && setEvalResult(null)}>
        <DialogContent className="sm:max-w-[600px] max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              AI Load Evaluation
            </DialogTitle>
            <DialogDescription>
              ML model recommendation for: {evalResult?.route}
            </DialogDescription>
          </DialogHeader>
          
          {evalResult && (
            <div className="space-y-4">
              {/* Decision Banner */}
              <div className={`p-4 rounded-lg border-2 ${
                evalResult.decision === 'ACCEPT' 
                  ? 'bg-emerald-500/10 border-emerald-500/30' 
                  : 'bg-red-500/10 border-red-500/30'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-2xl font-bold ${
                    evalResult.decision === 'ACCEPT' ? 'text-emerald-500' : 'text-red-500'
                  }`}>
                    {evalResult.decision === 'ACCEPT' ? '✓ ACCEPT LOAD' : '✗ SKIP LOAD'}
                  </span>
                  <Badge variant="outline" className="text-sm">
                    Score: {evalResult.confidence_score}/100
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">{evalResult.summary}</p>
              </div>

              {/* Route Economics */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold flex items-center gap-1">
                  <TrendingUp className="h-4 w-4 text-primary" /> Route Economics
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 rounded bg-muted/50">
                    <span className="text-[10px] text-muted-foreground uppercase">Load Price</span>
                    <p className="font-mono font-bold">₹{evalResult.economics?.price_inr?.toLocaleString()}</p>
                  </div>
                  <div className="p-2 rounded bg-muted/50">
                    <span className="text-[10px] text-muted-foreground uppercase">Total Cost</span>
                    <p className="font-mono font-bold">₹{evalResult.economics?.total_cost_inr?.toLocaleString()}</p>
                  </div>
                  <div className={`p-2 rounded ${evalResult.economics?.estimated_profit_inr > 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                    <span className="text-[10px] text-muted-foreground uppercase">Est. Profit</span>
                    <p className={`font-mono font-bold ${evalResult.economics?.estimated_profit_inr > 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                      ₹{evalResult.economics?.estimated_profit_inr?.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-2 rounded bg-muted/50">
                    <span className="text-[10px] text-muted-foreground uppercase">Profit Margin</span>
                    <p className="font-mono font-bold">{evalResult.economics?.profit_margin_percent}%</p>
                  </div>
                  <div className="p-2 rounded bg-muted/50">
                    <span className="text-[10px] text-muted-foreground uppercase">Rate</span>
                    <p className="font-mono font-bold">₹{evalResult.economics?.price_per_km}/km</p>
                  </div>
                  <div className="p-2 rounded bg-muted/50">
                    <span className="text-[10px] text-muted-foreground uppercase">Total Distance</span>
                    <p className="font-mono font-bold">{evalResult.economics?.total_distance_km} km</p>
                  </div>
                </div>
              </div>

              {/* ML Analysis */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold flex items-center gap-1">
                  <Sparkles className="h-4 w-4 text-primary" /> ML Model Analysis
                </h4>
                <div className="grid grid-cols-3 gap-2">
                  <div className="p-2 rounded bg-muted/50 text-center">
                    <span className="text-[10px] text-muted-foreground uppercase block">Delay Risk</span>
                    <p className="font-bold text-lg">{Math.round(evalResult.ml_analysis?.delay_probability * 100)}%</p>
                  </div>
                  <div className="p-2 rounded bg-muted/50 text-center">
                    <span className="text-[10px] text-muted-foreground uppercase block">Risk Level</span>
                    <p className={`font-bold text-sm ${
                      evalResult.ml_analysis?.risk_classification === 'High Risk' ? 'text-red-500' :
                      evalResult.ml_analysis?.risk_classification === 'Moderate Risk' ? 'text-amber-500' : 'text-emerald-500'
                    }`}>{evalResult.ml_analysis?.risk_classification}</p>
                  </div>
                  <div className="p-2 rounded bg-muted/50 text-center">
                    <span className="text-[10px] text-muted-foreground uppercase block">Fuel Rate</span>
                    <p className="font-bold text-sm">{evalResult.ml_analysis?.optimal_fuel_rate} L/100km</p>
                  </div>
                </div>
              </div>

              {/* Reasons */}
              {evalResult.reasons_accept?.length > 0 && (
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-emerald-500">Reasons to Accept</h4>
                  {evalResult.reasons_accept.map((r: string, i: number) => (
                    <p key={i} className="text-xs text-muted-foreground flex items-start gap-1">
                      <TrendingUp className="h-3 w-3 mt-0.5 text-emerald-500 shrink-0" /> {r}
                    </p>
                  ))}
                </div>
              )}
              {evalResult.reasons_reject?.length > 0 && (
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-red-500">Reasons to Skip</h4>
                  {evalResult.reasons_reject.map((r: string, i: number) => (
                    <p key={i} className="text-xs text-muted-foreground flex items-start gap-1">
                      <TrendingDown className="h-3 w-3 mt-0.5 text-red-500 shrink-0" /> {r}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </Layout>
  );
}
