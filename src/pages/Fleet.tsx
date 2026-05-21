import React from 'react';
import { motion } from 'framer-motion';
import { 
  Truck, 
  Fuel, 
  Activity, 
  BarChart3, 
  Search, 
  Filter, 
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Clock,
  PauseCircle
} from 'lucide-react';
import { Layout } from '@/components/Layout';
import { MetricCard, VehicleCard } from '@/components/Cards';
import { useRealtimeData } from '@/hooks/useRealtimeData';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { springPresets, fadeInUp, staggerContainer, staggerItem } from '@/lib/motion';

const Fleet: React.FC = () => {
  const { vehicles, metrics, isLoading, refresh } = useRealtimeData();
  const [searchQuery, setSearchQuery] = React.useState('');
  const [statusFilter, setStatusFilter] = React.useState<string>('all');

  const filteredVehicles = React.useMemo(() => {
    return vehicles.filter(vehicle => {
      const matchesSearch = 
        vehicle.plate_number.toLowerCase().includes(searchQuery.toLowerCase()) ||
        vehicle.model.toLowerCase().includes(searchQuery.toLowerCase()) ||
        vehicle.current_location.city.toLowerCase().includes(searchQuery.toLowerCase());
      
      const matchesStatus = statusFilter === 'all' || vehicle.status === statusFilter;
      
      return matchesSearch && matchesStatus;
    });
  }, [vehicles, searchQuery, statusFilter]);

  const activeCount = vehicles.filter(v => v.status === 'moving' || v.status === 'loading').length;
  const maintenanceCount = vehicles.filter(v => v.status === 'maintenance').length;
  const avgFuel = vehicles.length > 0 ? Math.round(vehicles.reduce((acc, v) => acc + v.fuel_level, 0) / vehicles.length) : 0;

  return (
    <Layout>
      <div className="space-y-8 p-6">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={springPresets.gentle}
          >
            <h1 className="text-3xl font-bold tracking-tight">Fleet Management</h1>
            <p className="text-muted-foreground mt-1 flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
              </span>
              Monitoring {vehicles.length} Indian registered commercial vehicles in real-time
            </p>
          </motion.div>

          <div className="flex items-center gap-3">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={refresh} 
              disabled={isLoading}
              className="bg-card"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh Data
            </Button>
            <Button size="sm" className="bg-primary hover:bg-primary/90 shadow-lg">
              Add New Vehicle
            </Button>
          </div>
        </div>

        {/* Metrics Grid */}
        <motion.div 
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          <motion.div variants={staggerItem}>
            <MetricCard
              title="Total Fleet"
              value={vehicles.length.toString()}
              change="+2 this month"
              icon={<Truck className="h-5 w-5 text-primary" />}
            />
          </motion.div>
          <motion.div variants={staggerItem}>
            <MetricCard
              title="In-Transit"
              value={activeCount.toString()}
              change={`${Math.round((activeCount/vehicles.length)*100)}% utilization`}
              icon={<Activity className="h-5 w-5 text-emerald-500" />}
            />
          </motion.div>
          <motion.div variants={staggerItem}>
            <MetricCard
              title="Avg Fuel Level"
              value={`${avgFuel}%`}
              change="-2.4% vs last week"
              icon={<Fuel className="h-5 w-5 text-amber-500" />}
            />
          </motion.div>
          <motion.div variants={staggerItem}>
            <MetricCard
              title="Maintenance"
              value={maintenanceCount.toString()}
              change="3 scheduled today"
              icon={<BarChart3 className="h-5 w-5 text-blue-500" />}
            />
          </motion.div>
        </motion.div>

        {/* Filters and Controls */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-card border border-border rounded-xl p-4 flex flex-col md:flex-row gap-4 items-center"
        >
          <div className="relative w-full md:w-96">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by plate number, model, or city..."
              className="pl-10 bg-background/50"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          
          <div className="flex items-center gap-3 w-full md:w-auto">
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-full md:w-[180px] bg-background/50">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Status Filter" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="idle">Idle</SelectItem>
                <SelectItem value="maintenance">Maintenance</SelectItem>
                <SelectItem value="delayed">Delayed</SelectItem>
              </SelectContent>
            </Select>

            <div className="flex gap-1 ml-auto">
              <div className="px-3 py-1.5 rounded-md border border-border bg-background flex items-center gap-2 text-xs font-medium">
                <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                Active
              </div>
              <div className="px-3 py-1.5 rounded-md border border-border bg-background flex items-center gap-2 text-xs font-medium">
                <PauseCircle className="h-3 w-3 text-slate-400" />
                Idle
              </div>
              <div className="px-3 py-1.5 rounded-md border border-border bg-background flex items-center gap-2 text-xs font-medium">
                <AlertTriangle className="h-3 w-3 text-amber-500" />
                Delayed
              </div>
              <div className="px-3 py-1.5 rounded-md border border-border bg-background flex items-center gap-2 text-xs font-medium">
                <Clock className="h-3 w-3 text-blue-500" />
                Service
              </div>
            </div>
          </div>
        </motion.div>

        {/* Fleet Grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-[320px] w-full rounded-2xl" />
            ))}
          </div>
        ) : filteredVehicles.length > 0 ? (
          <motion.div 
            variants={staggerContainer}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6"
          >
            {filteredVehicles.map((vehicle) => (
              <motion.div key={vehicle.id} variants={fadeInUp}>
                <VehicleCard vehicle={vehicle} />
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center py-24 text-center space-y-4">
            <div className="bg-muted rounded-full p-6">
              <Truck className="h-12 w-12 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-semibold">No vehicles found</h3>
              <p className="text-muted-foreground max-w-xs">
                Try adjusting your search or filter to find specific fleet units.
              </p>
            </div>
            <Button variant="outline" onClick={() => { setSearchQuery(''); setStatusFilter('all'); }}>
              Clear all filters
            </Button>
          </div>
        )}

        {/* Footer Statistics */}
        {!isLoading && filteredVehicles.length > 0 && (
          <div className="mt-12 pt-8 border-t border-border flex flex-col md:flex-row items-center justify-between text-sm text-muted-foreground gap-4">
            <p>© 2026 AI Path Logistics. All systems operational.</p>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-foreground">{metrics.utilizationRate}%</span> Fleet Utilization
              </div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-foreground">{metrics.onTimeDelivery}%</span> On-Time Rate
              </div>
              <div className="flex items-center gap-2">
                Last update: {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default Fleet;